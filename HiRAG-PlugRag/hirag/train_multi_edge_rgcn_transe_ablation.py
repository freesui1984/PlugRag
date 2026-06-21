# train_multi_edge_rgcn_transe_ablation.py
import os
import json
import random
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import RGCNConv
from torch_geometric.data import HeteroData
import torch_geometric

# ---------------------------
# 1. R-GCN + TransE 模型
# ---------------------------
class RGCNTransE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_layers=2, dropout=0.2, num_bases=None): 
        super().__init__()
        
        # 自动处理 num_bases 异常
        if num_bases is not None and (num_bases <= 0 or num_bases >= num_relations):
            num_bases = None
        
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # --- RGCN 层 (聚合邻居信息) ---
        # 第一层
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
        # 输出层
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases))

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(max(1, num_layers-1))])
        self.out_norm = torch.nn.LayerNorm(out_channels)

        # --- TransE 关系嵌入 ---
        # 关键：关系向量的维度必须与节点向量维度一致，才能做加减法
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, out_channels))
        nn.init.xavier_uniform_(self.relation_embedding)

    def forward(self, x, edge_index, edge_type):
        """前向传播：计算节点 Embedding"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                # LayerNorm
                ln = self.norms[min(i, len(self.norms)-1)]
                x = ln(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.out_norm(x)
        # TransE 通常需要对向量进行归一化，限制在单位球面上
        x = F.normalize(x, p=2, dim=1) 
        return x

# ---------------------------
# 2. 辅助工具函数
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def build_edge_index_and_types(data, device):
    """将 HeteroData 的边展平为 PyG 标准格式"""
    edge_index_list = []
    edge_type_list = []
    # 确保关系类型顺序固定
    rels = sorted(list(data.edge_types)) 
    
    for rel_id, rel in enumerate(rels):
        # HeteroData 存边的地方可能是 data[rel] 或者 data[src, rel, dst]
        # 这里做一种兼容处理
        try:
            ei = data[rel].edge_index
        except:
            continue
            
        if ei is None or ei.numel() == 0: continue
        edge_index_list.append(ei)
        edge_type_list.append(torch.full((ei.size(1),), rel_id, dtype=torch.long, device=device))
    
    if len(edge_index_list) == 0:
        return torch.empty((2,0), device=device), torch.empty((0,), device=device), rels
    
    edge_index_all = torch.cat(edge_index_list, dim=1).to(device)
    edge_type_all = torch.cat(edge_type_list, dim=0).to(device)
    return edge_index_all, edge_type_all, rels

def split_edges(num_edges, train_ratio=0.85, val_ratio=0.05):
    """随机划分数据集"""
    indices = torch.randperm(num_edges)
    n_train = int(num_edges * train_ratio)
    n_val = int(num_edges * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return train_idx, val_idx, test_idx

# ---------------------------
# 3. 评估函数 (计算 MRR)
# ---------------------------
@torch.no_grad()
def evaluate_mrr(model, x, edge_index, edge_type, eval_idx, device, batch_size=100):
    """
    使用 TransE 评分函数计算 MRR
    Score = -|| h + r - t || (距离越小，得分越高)
    """
    model.eval()
    
    # 1. 获取全图节点的 Embedding (作为候选池)
    all_emb = model(x, edge_index, edge_type) # [N, Dim]
    rel_emb = model.relation_embedding        # [R, Dim]
    
    # 准备评估数据
    src = edge_index[0, eval_idx]
    dst = edge_index[1, eval_idx]
    rel = edge_type[eval_idx]
    
    ranks = []
    num_eval = eval_idx.size(0)
    
    # 分批计算
    for i in range(0, num_eval, batch_size):
        b_src = src[i:i+batch_size]
        b_dst = dst[i:i+batch_size]
        b_rel = rel[i:i+batch_size]
        
        # 构造 Query: h + r
        h = all_emb[b_src]
        r = rel_emb[b_rel]
        target_emb = all_emb[b_dst]
        
        query = h + r # [B, Dim]
        
        # 计算 query 到所有节点的距离 (L2 Norm)
        # 使用 cdist 高效计算: shape [B, N]
        dists = torch.cdist(query, all_emb, p=2)
        
        # 获取真实目标节点的距离
        # dists[row, b_dst[row]]
        true_dists = dists.gather(1, b_dst.view(-1, 1))
        
        # 排名逻辑：有多少个节点的距离 < 真实距离？
        # Rank = count(d < true_d) + 1
        # 这种比较方式对 TransE (基于距离) 是正确的
        rank = (dists < true_dists).sum(dim=1).float() + 1
        ranks.append(rank)
        
    if len(ranks) == 0:
        return 0.0, 0.0, 0.0, 0.0

    ranks = torch.cat(ranks)
    mrr = (1.0 / ranks).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    
    return mrr, hits1, hits3, hits10

# ---------------------------
# 4. 训练主循环
# ---------------------------
def train_pipeline(cfg, device):
    set_seed(cfg.SEED)
    
    # --- 加载数据 ---
    if not os.path.exists(cfg.INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {cfg.INPUT_FILE}")
        return

    print(f"📥 Loading data from {cfg.INPUT_FILE}...")
    data = torch.load(cfg.INPUT_FILE, map_location="cpu", weights_only=False)
    
    node_type = 'entity' if 'entity' in data.node_types else data.node_types[0]
    x = data[node_type].x.to(device)
    x_truth_semantic = F.normalize(x.detach().clone(), p=2, dim=1) # 原始语义锚点
    
    edge_index, edge_type, rels = build_edge_index_and_types(data, device)
    num_nodes = x.size(0)
    num_relations = len(rels)
    
    # 数据划分
    train_idx, val_idx, test_idx = split_edges(edge_index.size(1))
    print(f"📊 Graph Info: {num_nodes} nodes, {num_relations} relation types.")
    print(f"📊 Dataset Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # --- 初始化模型 ---
    model = RGCNTransE(
        in_channels=x.size(1),
        hidden_channels=cfg.HIDDEN_DIM,
        out_channels=x.size(1), # 输出维度保持与输入一致，方便对比
        num_relations=num_relations,
        num_layers=cfg.NUM_LAYERS,
        dropout=0.2,
        num_bases=cfg.NUM_BASES
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
    # 学习率调度：基于 MRR 调整 (mode='max')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=20, verbose=True
    )
    
    # 定义损失函数
    mse_loss_fn = nn.MSELoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=cfg.MARGIN, p=2)

    print(f"🚀 Start Training (Hybrid TransE)... Alpha={cfg.ALPHA}, Beta={cfg.BETA}")
    
    best_val_mrr = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # 1. 前向传播
        h = model(x, edge_index, edge_type)
        
        # 2. 计算语义损失 (Anchor)
        loss_anchor = mse_loss_fn(h, x_truth_semantic)
        
        # 3. 计算结构损失 (TransE Triplet) - 只用训练集边
        if len(train_idx) > 0:
            # 随机采样一个 Batch 进行训练 (全量训练太慢)
            # 如果显存够大，可以用 len(train_idx)，不够就用 cfg.BATCH_SIZE
            batch_size = min(len(train_idx), 10000) 
            perm = torch.randperm(len(train_idx))[:batch_size]
            batch_idx = train_idx[perm]
            
            src = edge_index[0, batch_idx]
            dst = edge_index[1, batch_idx]
            rel = edge_type[batch_idx]
            
            # 负采样 (替换尾节点)
            neg_dst = torch.randint(0, num_nodes, (batch_size,), device=device)
            
            # 获取向量
            h_src = h[src]
            h_dst = h[dst]
            h_neg = h[neg_dst]
            r_emb = model.relation_embedding[rel]
            
            # TransE 逻辑: Anchor = h + r, Positive = t
            # 我们希望 ||(h+r) - t|| 越小越好
            # Triplet Loss 会自动拉近 (anchor, positive) 推远 (anchor, negative)
            anchor_term = h_src + r_emb
            loss_link = triplet_loss_fn(anchor_term, h_dst, h_neg)
        else:
            loss_link = torch.tensor(0.0, device=device)
            
        # 4. 总损失
        loss = (cfg.ALPHA * cfg.MSE_SCALE * loss_anchor) + (cfg.BETA * loss_link)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # --- 评估与监控 (每 10 轮) ---
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            # 计算验证集 MRR
            val_mrr, v_h1, v_h3, v_h10 = evaluate_mrr(model, x, edge_index, edge_type, val_idx, device)
            
            # 更新 LR
            scheduler.step(val_mrr)
            
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} (Link: {loss_link.item():.4f}) | "
                  f"Val MRR: {val_mrr:.4f} (H@10: {v_h10:.2f}) | Time: {elapsed:.1f}s")
            start_time = time.time()
            
            # 早停逻辑
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                epochs_no_improve = 0
                # 保存最佳模型
                out_dict = {
                    'embeddings': h.detach().cpu(), 
                    'rel_embeddings': model.relation_embedding.detach().cpu(),
                    'rel_names': rels,
                    'mrr': val_mrr
                }
                torch.save(out_dict, cfg.OUTPUT_FILE + ".best")
                # print(f"  -> New Best Model Saved (MRR: {val_mrr:.4f})")
            else:
                epochs_no_improve += 10 # 每10轮检测一次
                
            if epochs_no_improve >= cfg.EARLY_STOPPING:
                print(f"\n🛑 Early stopping triggered. Best Val MRR: {best_val_mrr:.4f}")
                break

    # --- 结束处理 ---
    if os.path.exists(cfg.OUTPUT_FILE + ".best"):
        print("💾 Loading best model for testing...")
        best_checkpoint = torch.load(cfg.OUTPUT_FILE + ".best")
        torch.save(best_checkpoint, cfg.OUTPUT_FILE) # 覆盖为最终文件
        os.remove(cfg.OUTPUT_FILE + ".best")
        
        # 跑测试集
        # 注意：这里我们无法恢复模型权重(state_dict)，但我们保存了最优的 embeddings
        # 对于 Transductive 任务，我们直接用这组 embeddings 进行推理即可
        # 重新实例化一个 Dummy Model 来承载参数进行评估有点麻烦，
        # 我们这里直接手动计算 Test MRR (如果需要)
        print(f"✅ Training Finished. Best Embeddings saved to {cfg.OUTPUT_FILE}")
        print(f"🏆 Best Val MRR: {best_val_mrr:.4f}")
    else:
        print("⚠️ No model saved.")

# ---------------------------
# 5. 配置加载与入口
# ---------------------------
class TrainingConfig:
    def __init__(self):
        # 默认值
        self.INPUT_FILE = "rgcn_data.pt"
        self.OUTPUT_FILE = "rgcn_embeddings.pt"
        self.HIDDEN_DIM = 768
        self.NUM_LAYERS = 2
        self.NUM_BASES = 20
        self.LEARNING_RATE = 0.001
        self.EPOCHS = 2000
        self.EARLY_STOPPING = 100
        self.ALPHA = 0.1
        self.BETA = 1.0
        self.MSE_SCALE = 50.0
        self.MARGIN = 1.0
        self.SEED = 42
        
        # 读取 yaml 覆盖默认值
        config_path = os.environ.get("PLUGRAG_CONFIG", "config.example.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f)
                tc = yaml_cfg.get(
                    "multi_edge_rgcn_transe", {}
                ).get("training", yaml_cfg.get("rgcn", {}).get("training", {}))
                self.INPUT_FILE = tc.get("input_pt", self.INPUT_FILE)
                self.OUTPUT_FILE = tc.get("output_pt", self.OUTPUT_FILE)
                self.HIDDEN_DIM = int(tc.get("hidden_dim", self.HIDDEN_DIM))
                self.NUM_LAYERS = int(tc.get("num_layers", self.NUM_LAYERS))
                self.NUM_BASES = int(tc.get("num_bases", self.NUM_BASES))
                self.LEARNING_RATE = float(tc.get("learning_rate", self.LEARNING_RATE))
                self.EPOCHS = int(tc.get("epochs", self.EPOCHS))
                self.EARLY_STOPPING = int(tc.get("early_stopping", self.EARLY_STOPPING))
                self.ALPHA = float(tc.get("alpha", self.ALPHA))
                self.BETA = float(tc.get("beta", self.BETA))
                self.MSE_SCALE = float(tc.get("mse_scale", self.MSE_SCALE))
                self.MARGIN = float(tc.get("margin", self.MARGIN))
                self.SEED = int(tc.get("seed", self.SEED))

if __name__ == "__main__":
    cfg = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 注册 PyG 类
    try:
        from torch.serialization import add_safe_globals
        from torch_geometric.data.storage import BaseStorage
        add_safe_globals([BaseStorage, HeteroData, torch_geometric.data.Data])
    except: pass

    train_pipeline(cfg, device)

    mapping_in = os.path.splitext(cfg.INPUT_FILE)[0] + "_mapping.json"
    mapping_out = os.path.splitext(cfg.OUTPUT_FILE)[0] + "_mapping.json"
    if os.path.exists(mapping_in):
        with open(mapping_in, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        with open(mapping_out, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"🔖 Mapping JSON saved to: {mapping_out}")
