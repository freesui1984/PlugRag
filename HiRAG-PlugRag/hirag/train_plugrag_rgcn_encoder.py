# train_plugrag_rgcn_encoder.py
import argparse
import json
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import HeteroData
import torch_geometric
import torch, random, numpy as np
import yaml
from torch.utils.checkpoint import checkpoint
# ---------------------------
# 1. R-GCN 编码器 (不变)
# ---------------------------
class RGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_layers=2, dropout=0.2, num_bases=30): 
        """
        R-GCN 编码器，已启用 Basis Decomposition (num_bases) 来处理不均衡的多关系。
        
        参数:
        ...
        num_relations (int): 关系类型的总数 (例如 48)。
        num_bases (int): 
            要使用的“基础矩阵”的数量。
            这是一种正则化技巧，用于解决关系不均衡问题。
            它必须小于 num_relations。 
            如果设为 None 或 0，则退化为朴素的 R-GCN。
        """
        super().__init__()
        assert num_layers >= 2
        
        # ⭐️ [修改] 确保 num_bases 是一个合理的值
        if num_bases is not None and num_bases >= num_relations:
            print(f"警告: num_bases ({num_bases}) 大于或等于 num_relations ({num_relations}). "
                  f"将退化为朴素 R-GCN。建议设置 num_bases < num_relations。")
            num_bases = None
        elif num_bases is None or num_bases <= 0: # 修正为 <= 0
             print("信息: 未指定 num_bases 或为 0，使用朴素 R-GCN (每种关系一个独立权重)。")
             num_bases = None # 确保 PyG 接收到的是 None 而不是 0
        else:
            print(f"信息: R-GCN 启用 Basis Decomposition，使用 {num_bases} 个基础矩阵。")

        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # --- ⭐️ [核心修改] ---
        # 1. 第一层: 
        #    将 num_bases 传递给 RGCNConv
        self.convs.append(RGCNConv(
            in_channels, 
            hidden_channels, 
            num_relations,
            num_bases=num_bases 
        ))

        # 2. 中间层:
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(
                hidden_channels, 
                hidden_channels, 
                num_relations,
                num_bases=num_bases
            ))

        # 3. 最后一层:
        self.convs.append(RGCNConv(
            hidden_channels, 
            out_channels, 
            num_relations,
            num_bases=num_bases
        ))
        # --- ⭐️ [修改结束] ---

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(max(1, num_layers-1))])
        self.out_norm = torch.nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_type):
        # 必须定义一个简单的内部函数来给 checkpoint 调用
        def run_layer(layer, x, edge_index, edge_type):
            return layer(x, edge_index, edge_type)

        for i, conv in enumerate(self.convs):
            # === 核心修改 ===
            # 仅在训练阶段且 x 需要梯度时开启 checkpoint
            if self.training and x.requires_grad:
                # use_reentrant=False 更稳定
                x = checkpoint(run_layer, conv, x, edge_index, edge_type, use_reentrant=False)
            else:
                x = conv(x, edge_index, edge_type)
            # ===============
            if i != self.num_layers - 1:
                x = F.relu(x)
                ln = self.norms[min(i, len(self.norms)-1)]
                x = ln(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_norm(x)
        x = F.normalize(x, p=2, dim=1) 
        return x

# ---------------------------
# 2. 辅助函数 (不变)
# ---------------------------
def set_seed(seed=42):
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_edge_index_and_types(data, device):
    edge_index_list = []
    edge_type_list = []
    rels = sorted(list(data.edge_types)) 
    for rel_id, rel in enumerate(rels):
        v = data[rel]
        if not hasattr(v, 'edge_index'): continue
        ei = v.edge_index
        if ei is None or ei.numel() == 0: continue
        edge_index_list.append(ei)
        edge_type_list.append(torch.full((ei.size(1),), rel_id, dtype=torch.long, device=device))
    if len(edge_index_list) == 0:
        edge_index_all = torch.empty((2,0), dtype=torch.long, device=device)
        edge_type_all = torch.empty((0,), dtype=torch.long, device=device)
    else:
        edge_index_all = torch.cat(edge_index_list, dim=1).to(device)
        edge_type_all = torch.cat(edge_type_list, dim=0).to(device)
    return edge_index_all, edge_type_all, rels

# ---------------------------
# 3. "混合损失" 训练函数 (⭐️ [已修复] ⭐️)
# ---------------------------
def train_hybrid(data, hidden, lr, epochs, device, output_path,
                 alpha=0.5, beta=0.5, margin=1.0, 
                 mse_scale=500.0, 
                 weight_decay=1e-5, 
                 scheduler_patience=20,
                 early_stopping_patience=20, 
                 num_layers=3, 
                 seed=42,
                 num_bases=None): # ⭐️ [修复] 接收 num_bases 参数

    set_seed(seed)
    
    # --- 1. 数据准备 ---
    node_type = 'entity' if 'entity' in data.node_types else data.node_types[0]
    x = data[node_type].x.to(device) 
    num_nodes = x.size(0)
    x_truth_semantic = F.normalize(x.detach().clone(), p=2, dim=1) 
    edge_index, edge_type, rels = build_edge_index_and_types(data, device)
    num_relations = len(rels)
    print(f"✅ 数据准备完毕: {num_nodes} 节点, {edge_index.size(1)} 边, {num_relations} 关系。")

    # --- 2. 模型准备 ---
    in_channels = x.size(1)
    out_dim = in_channels 
    print(f"✅ 模型准备: In/Out Dim = {in_channels}, Hidden Dim = {hidden}, Layers={num_layers}")
    
    # ⭐️ [修复] 确保 num_bases 正确传入 RGCNEncoder
    model = RGCNEncoder(in_channels=in_channels,
                        hidden_channels=hidden,
                        out_channels=out_dim,
                        num_relations=num_relations,
                        num_layers=num_layers,
                        num_bases=num_bases, # ⭐️ [修复] 将 num_bases 传入
                        dropout=0.2).to(device)
                        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience)

    # --- 3. 损失函数定义 ---
    mse_loss_fn = nn.MSELoss() 
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2.0) 

    print(f"🚀 开始训练 (混合策略)... Alpha={alpha}, Beta={beta}, Margin={margin}, MSE_Scale={mse_scale}")
    print(f"   (学习率调度耐心: {scheduler_patience} 轮, 早停耐心: {early_stopping_patience} 轮)")
    
    # --- 4. 训练循环 ---
    best_loss = float('inf')
    epochs_no_improve = 0 
    start_time = time.time()
    
    # ⭐️ [优化] 仅在边数 > 0 时才定义正例
    if edge_index.size(1) > 0:
        positive_anchors_idx = edge_index[0]
        positive_partners_idx = edge_index[1]
    else:
        # 如果没有边，Triplet Loss 将始终为 0
        positive_anchors_idx = torch.tensor([], dtype=torch.long, device=device)
        positive_partners_idx = torch.tensor([], dtype=torch.long, device=device)

    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        h = model(x, edge_index, edge_type)
        
        # 2. [核心] 计算 L_anchor (锚定损失)
        loss_anchor = mse_loss_fn(h, x_truth_semantic)
        
        # 3. [核心] 计算 L_explore (探索损失)
        # ⭐️ [优化] 仅在有边的情况下才计算 Triplet Loss
        if positive_anchors_idx.numel() > 0:
            h_anchors = h[positive_anchors_idx]
            h_positives = h[positive_partners_idx]
            negative_indices = torch.randint(0, num_nodes, (positive_anchors_idx.size(0),), device=device)
            mask = (negative_indices == positive_partners_idx)
            negative_indices[mask] = (negative_indices[mask] + 1) % num_nodes
            h_negatives = h[negative_indices]
            loss_explore = triplet_loss_fn(h_anchors, h_positives, h_negatives)
        else:
            loss_explore = torch.tensor(0.0, device=device) # 如果没有边，探索损失为 0
        
        scaled_loss_anchor = mse_scale * loss_anchor
        loss = (alpha * scaled_loss_anchor) + (beta * loss_explore)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f} | L_Anchor(Scaled): {scaled_loss_anchor.item():.6f} | L_Explore(Triplet): {loss_explore.item():.6f} | Time(s): {elapsed:.1f}")
            start_time = time.time()

        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_no_improve = 0 
            out_dict = {'embeddings': h.detach().cpu(), 'rel_names': rels}
            torch.save(out_dict, output_path + ".best.pt")
        else:
            epochs_no_improve += 1 

        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            print(f"\n[早停] 损失在 {early_stopping_patience} 轮内没有改善，于第 {epoch} 轮停止训练。")
            break 
            
    # --- 训练结束，保存最终模型 ---
    if os.path.exists(output_path + ".best.pt"):
        print(f"💾 加载最佳嵌入 (Loss={best_loss:.6f}) 并保存到最终路径...")
        out_dict = torch.load(output_path + ".best.pt", map_location='cpu')
        torch.save(out_dict, output_path)
        final_embeddings = out_dict['embeddings']
        print(f"💾 最终混合嵌入已保存到: {output_path} (shape: {tuple(final_embeddings.shape)})")
        os.remove(output_path + ".best.pt") 
        return final_embeddings
    else:
        print(f"⚠️ 训练期间未保存任何最佳模型 (可能 epochs=0 或训练失败)。")
        return None

# ---------------------------
# 5. 主函数 (CLI 入口) (⭐️ [已修复] ⭐️)
# ---------------------------

# ⭐️ [新增] 配置类，用于在代码中设置参数
class TrainingConfig:
    def __init__(self):
        config_path = os.environ.get("PLUGRAG_CONFIG", os.path.join(os.getcwd(), "config.example.yaml"))
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tc = cfg.get("rgcn", {}).get("training", {})
        self.INPUT_FILE = tc.get("input_pt", "work_dir_data_02/rgcn_data.pt")
        self.OUTPUT_FILE = tc.get("output_pt", "work_dir_data_02/rgcn_embeddings.pt")
        self.HIDDEN_DIM = int(tc.get("hidden_dim", 512))
        self.NUM_LAYERS = int(tc.get("num_layers", 3))
        self.NUM_BASES = int(tc.get("num_bases", 20))
        self.LEARNING_RATE = float(tc.get("learning_rate", 1e-4))
        self.EPOCHS = int(tc.get("epochs", 2000))
        self.EARLY_STOPPING = int(tc.get("early_stopping", 20))
        self.ALPHA = float(tc.get("alpha", 0.8))
        self.BETA = float(tc.get("beta", 0.2))
        self.MSE_SCALE = float(tc.get("mse_scale", 500.0))
        self.MARGIN = float(tc.get("margin", 1.0))
        self.SEED = int(tc.get("seed", 42))

def main():
    # ⭐️ [修改] 不再使用 argparse，直接从 TrainingConfig 加载配置
    cfg = TrainingConfig()

    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        from torch.serialization import add_safe_globals
        from torch_geometric.data.storage import BaseStorage
        add_safe_globals([BaseStorage, HeteroData, torch_geometric.data.Data])
    except Exception: pass

    print("Loading HeteroData from:", cfg.INPUT_FILE)
    data = torch.load(cfg.INPUT_FILE, map_location="cpu", weights_only=False) 
    
    # 运行训练
    train_hybrid(data, 
                 hidden=cfg.HIDDEN_DIM, 
                 lr=cfg.LEARNING_RATE, 
                 epochs=cfg.EPOCHS, 
                 device=device, 
                 output_path=cfg.OUTPUT_FILE,
                 alpha=cfg.ALPHA, 
                 beta=cfg.BETA, 
                 margin=cfg.MARGIN,
                 mse_scale=cfg.MSE_SCALE,
                 early_stopping_patience=cfg.EARLY_STOPPING,
                 num_layers=cfg.NUM_LAYERS,
                 num_bases=cfg.NUM_BASES, # ⭐️ [修复] 传入 num_bases
                 seed=cfg.SEED)
    
    # 自动复制映射文件
    mapping_json_candidate = os.path.splitext(cfg.INPUT_FILE)[0] + "_mapping.json"
    mapping_out = os.path.splitext(cfg.OUTPUT_FILE)[0] + "_mapping.json"
    if os.path.exists(mapping_json_candidate):
        print(f"🔖 正在复制映射文件...")
        with open(mapping_json_candidate, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        with open(mapping_out, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"🔖 已保存映射 JSON 到: {mapping_out}")


if __name__ == "__main__":
    main()
