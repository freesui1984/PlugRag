# train_simple_gcn_ablation.py
import argparse
import json
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# 【关键修改】导入 GCNConv 而不是 RGCNConv
from torch_geometric.nn import GCNConv 
from torch_geometric.data import Data
import torch_geometric
import numpy as np
import yaml

# ---------------------------
# 1. GCN 编码器 (同质图版本)
# ---------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2): 
        """
        标准 GCN 编码器
        区别：
        1. 不需要 num_relations
        2. 不需要 num_bases
        """
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # 1. 第一层
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # 2. 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # 3. 最后一层
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(max(1, num_layers-1))])
        self.out_norm = torch.nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        # 【关键修改】forward 不再接收 edge_type
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index) # GCN 只需要 x 和 edge_index
            
            if i != self.num_layers - 1:
                x = F.relu(x)
                ln = self.norms[min(i, len(self.norms)-1)]
                x = ln(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.out_norm(x)
        x = F.normalize(x, p=2, dim=1) 
        return x

# ---------------------------
# 2. 辅助函数
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ---------------------------
# 3. 混合损失训练函数 (GCN版)
# ---------------------------
def train_hybrid_gcn(data, hidden, lr, epochs, device, output_path,
                     alpha=0.8, beta=0.2, margin=1.0, 
                     mse_scale=50.0, 
                     weight_decay=1e-5, 
                     scheduler_patience=20,
                     early_stopping_patience=20, 
                     num_layers=3, 
                     seed=42):

    set_seed(seed)
    
    # --- 1. 数据准备 ---
    # GCN 数据集是 Homogeneous Data，直接取 x 和 edge_index
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    num_nodes = x.size(0)
    # 原始语义作为锚点
    x_truth_semantic = F.normalize(x.detach().clone(), p=2, dim=1) 
    
    print(f"✅ GCN数据准备完毕: {num_nodes} 节点, {edge_index.size(1)} 边。")

    # --- 2. 模型准备 ---
    in_channels = x.size(1)
    out_dim = in_channels 
    
    # 【关键修改】实例化 GCNEncoder
    model = GCNEncoder(in_channels=in_channels,
                       hidden_channels=hidden,
                       out_channels=out_dim,
                       num_layers=num_layers,
                       dropout=0.2).to(device)
                        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience)

    # --- 3. 损失函数 ---
    mse_loss_fn = nn.MSELoss() 
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2.0) 

    print(f"🚀 开始 GCN 训练... Alpha={alpha}, Beta={beta}")
    
    # --- 4. 训练循环 ---
    best_loss = float('inf')
    epochs_no_improve = 0 
    
    # 准备正例索引
    if edge_index.size(1) > 0:
        positive_anchors_idx = edge_index[0]
        positive_partners_idx = edge_index[1]
    else:
        positive_anchors_idx = torch.tensor([], device=device)
        positive_partners_idx = torch.tensor([], device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 【关键修改】前向传播只传两个参数
        h = model(x, edge_index)
        
        # --- Loss 1: 语义锚点 ---
        loss_anchor = mse_loss_fn(h, x_truth_semantic)
        
        # --- Loss 2: 结构探索 (Triplet) ---
        if positive_anchors_idx.numel() > 0:
            h_anchors = h[positive_anchors_idx]
            h_positives = h[positive_partners_idx]
            
            # 随机负采样
            negative_indices = torch.randint(0, num_nodes, (positive_anchors_idx.size(0),), device=device)
            # 简单的防撞车逻辑
            mask = (negative_indices == positive_partners_idx)
            negative_indices[mask] = (negative_indices[mask] + 1) % num_nodes
            
            h_negatives = h[negative_indices]
            
            loss_explore = triplet_loss_fn(h_anchors, h_positives, h_negatives)
        else:
            loss_explore = torch.tensor(0.0, device=device)
        
        # 联合损失
        scaled_loss_anchor = mse_scale * loss_anchor
        loss = (alpha * scaled_loss_anchor) + (beta * loss_explore)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f} | L_Anchor: {scaled_loss_anchor.item():.4f} | L_Struct: {loss_explore.item():.4f}")

        # 早停与保存
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_no_improve = 0 
            # 保存时去掉 rel_names，因为 GCN 没有关系名
            out_dict = {'embeddings': h.detach().cpu()}
            torch.save(out_dict, output_path + ".best.pt")
        else:
            epochs_no_improve += 1 

        if epochs_no_improve >= early_stopping_patience:
            print(f"\n[早停] 第 {epoch} 轮停止。Best Loss: {best_loss:.6f}")
            break 
            
    # 加载最佳模型
    if os.path.exists(output_path + ".best.pt"):
        out_dict = torch.load(output_path + ".best.pt", map_location='cpu')
        torch.save(out_dict, output_path)
        os.remove(output_path + ".best.pt") 
        print(f"💾 最终 GCN 嵌入已保存: {output_path}")
    else:
        print("⚠️ 训练失败，未保存模型。")

# ---------------------------
# 4. 配置类与主函数
# ---------------------------
class TrainingConfig:
    def __init__(self):
        config_path = os.environ.get("PLUGRAG_CONFIG", os.path.join(os.getcwd(), "config.example.yaml"))
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # 这里为了方便，我们直接读取 gcn 下的配置，如果没有就读取 rgcn 下的并覆盖
        # 优先读取 cfg['gcn']['training']，如果没有，退化读取 cfg['rgcn']['training']
        tc = cfg.get("gcn", {}).get("training", cfg.get("rgcn", {}).get("training", {}))
        
        self.INPUT_FILE = tc.get("input_pt", "work_dir_data_mix/gcn_data_gcn.pt")
        self.OUTPUT_FILE = tc.get("output_pt", "work_dir_data_mix/gcn_embeddings.pt")
        self.HIDDEN_DIM = int(tc.get("hidden_dim", 512))
        self.NUM_LAYERS = int(tc.get("num_layers", 3))
        self.LEARNING_RATE = float(tc.get("learning_rate", 0.001))
        self.EPOCHS = int(tc.get("epochs", 2000))
        self.EARLY_STOPPING = int(tc.get("early_stopping", 20))
        self.ALPHA = float(tc.get("alpha", 0.8))
        self.BETA = float(tc.get("beta", 0.2))
        self.MSE_SCALE = float(tc.get("mse_scale", 50.0))
        self.MARGIN = float(tc.get("margin", 1.0))
        self.SEED = int(tc.get("seed", 42))

def main():
    cfg = TrainingConfig()
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 使用设备: {device}")
    print(f"🔹 配置文件: {cfg.INPUT_FILE}")

    try:
        # 注册 safe globals 以防 load 报错
        from torch_geometric.data import Data
        from torch.serialization import add_safe_globals
        add_safe_globals([Data])
    except: pass

    if not os.path.exists(cfg.INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {cfg.INPUT_FILE}")
        return

    # 加载数据
    data = torch.load(cfg.INPUT_FILE, map_location="cpu")
    
    # 运行训练
    train_hybrid_gcn(
        data, 
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
        seed=cfg.SEED
    )
    
    # 复制 mapping 文件
    mapping_in = cfg.INPUT_FILE.replace(".pt", "_mapping.json")
    mapping_out = cfg.OUTPUT_FILE.replace(".pt", "_mapping.json")
    if os.path.exists(mapping_in):
        with open(mapping_in, 'r', encoding='utf-8') as f:
            d = json.load(f)
        with open(mapping_out, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)
        print(f"🔖 映射文件已复制: {mapping_out}")

if __name__ == "__main__":
    main()
