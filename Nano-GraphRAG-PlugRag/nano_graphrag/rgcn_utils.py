import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, RGCNConv
from torch_geometric.data import Data, HeteroData
from torch.utils.checkpoint import checkpoint
from ._utils import logger, compute_mdhash_id


def _iter_graph_edges(graph):
    if hasattr(graph, "is_multigraph") and graph.is_multigraph():
        for u, v, key, attr in graph.edges(data=True, keys=True):
            yield u, v, key, attr or {}
    else:
        for u, v, attr in graph.edges(data=True):
            yield u, v, None, attr or {}


def _relation_label(edge_key, edge_attr):
    raw = (
        edge_attr.get("relation_type")
        or edge_attr.get("relation")
        or edge_attr.get("label")
        or edge_key
        or "RELATED_TO"
    )
    return str(raw).strip().upper().replace(" ", "_")

# ---------------------------
# 1. R-GCN 编码器
# ---------------------------
class RGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_layers=2, dropout=0.2, num_bases=30): 
        super().__init__()
        assert num_layers >= 2
        
        if num_bases is not None and num_bases >= num_relations:
            logger.warning(f"num_bases ({num_bases}) >= num_relations ({num_relations}). Fallback to naive RGCN.")
            num_bases = None
        elif num_bases is None or num_bases <= 0:
             num_bases = None
        else:
            logger.info(f"RGCN enabled Basis Decomposition with {num_bases} bases.")

        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases))

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(max(1, num_layers-1))])
        self.out_norm = torch.nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_type):
        def run_layer(layer, x, edge_index, edge_type):
            return layer(x, edge_index, edge_type)

        for i, conv in enumerate(self.convs):
            if self.training and x.requires_grad:
                x = checkpoint(run_layer, conv, x, edge_index, edge_type, use_reentrant=False)
            else:
                x = conv(x, edge_index, edge_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                ln = self.norms[min(i, len(self.norms)-1)]
                x = ln(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_norm(x)
        x = F.normalize(x, p=2, dim=1) 
        return x


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        assert num_layers >= 2

        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_channels) for _ in range(max(1, num_layers - 1))]
        )
        self.out_norm = torch.nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.norms[min(i, len(self.norms) - 1)](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_norm(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class RGCNTransE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_relations,
        num_layers=2,
        dropout=0.2,
        num_bases=None,
    ):
        super().__init__()
        assert num_layers >= 2

        if num_bases is not None and (num_bases <= 0 or num_bases >= num_relations):
            num_bases = None

        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases)
            )
        self.convs.append(
            RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases)
        )

        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_channels) for _ in range(max(1, num_layers - 1))]
        )
        self.out_norm = torch.nn.LayerNorm(out_channels)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, out_channels))
        nn.init.xavier_uniform_(self.relation_embedding)

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.norms[min(i, len(self.norms) - 1)](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_norm(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def set_seed(seed=42):
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
        return torch.empty((2,0), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device), rels
    
    return torch.cat(edge_index_list, dim=1).to(device), torch.cat(edge_type_list, dim=0).to(device), rels

async def convert_to_rgcn_data(graph_storage, entity_vdb, output_pt_path):
    logger.info("Converting graph to RGCN format...")
    os.makedirs(os.path.dirname(output_pt_path) or ".", exist_ok=True)
    vdb_embeddings = await entity_vdb.get_all_embeddings()
    node_ids = list(graph_storage._graph.nodes)
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    node_features = []
    for node_id in node_ids:
        vdb_id = compute_mdhash_id(node_id, prefix="ent-")
        if vdb_id in vdb_embeddings:
            node_features.append(vdb_embeddings[vdb_id])
        else:
            dim = next(iter(vdb_embeddings.values())).shape[0] if vdb_embeddings else 1536
            node_features.append(np.zeros(dim))
    
    node_features = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index_dict = {}
    relation_types = set()
    
    for u, v, key, data in _iter_graph_edges(graph_storage._graph):
        rel_type = _relation_label(key, data)
        relation_types.add(rel_type)
        rel_key = ('entity', rel_type, 'entity')
        if rel_key not in edge_index_dict:
            edge_index_dict[rel_key] = [[], []]
        edge_index_dict[rel_key][0].append(node_to_idx[u])
        edge_index_dict[rel_key][1].append(node_to_idx[v])

    data = HeteroData()
    data['entity'].x = node_features
    for rel_key, (src, dst) in edge_index_dict.items():
        data[rel_key].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    torch.save(data, output_pt_path)
    mapping_path = os.path.splitext(output_pt_path)[0] + "_mapping.json"
    mappings = {"node_to_idx": node_to_idx, "idx_to_node": {i: n for i, n in enumerate(node_ids)}, "relation_types": sorted(list(relation_types))}
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    return data, mappings


async def convert_to_multi_edge_rgcn_data(graph_storage, entity_vdb, output_pt_path):
    return await convert_to_rgcn_data(graph_storage, entity_vdb, output_pt_path)


async def convert_to_gcn_data(graph_storage, entity_vdb, output_pt_path):
    logger.info("Converting graph to homogeneous GCN format...")
    os.makedirs(os.path.dirname(output_pt_path) or ".", exist_ok=True)
    vdb_embeddings = await entity_vdb.get_all_embeddings()
    node_ids = list(graph_storage._graph.nodes)
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    node_features = []
    for node_id in node_ids:
        vdb_id = compute_mdhash_id(node_id, prefix="ent-")
        if vdb_id in vdb_embeddings:
            node_features.append(vdb_embeddings[vdb_id])
        else:
            dim = next(iter(vdb_embeddings.values())).shape[0] if vdb_embeddings else 1536
            node_features.append(np.zeros(dim))

    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_set = set()
    for u, v in graph_storage._graph.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_set.add((node_to_idx[u], node_to_idx[v]))

    if edge_set:
        src, dst = zip(*sorted(edge_set))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = len(node_ids)
    torch.save(data, output_pt_path)

    mapping_path = os.path.splitext(output_pt_path)[0] + "_mapping.json"
    mappings = {
        "node_to_idx": node_to_idx,
        "idx_to_node": {i: n for i, n in enumerate(node_ids)},
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)
    return data, mappings

def train_hybrid(data, hidden, lr, epochs, device, output_path,
                 alpha=0.5, beta=0.5, margin=1.0, mse_scale=500.0, 
                 weight_decay=1e-5, scheduler_patience=20,
                 early_stopping_patience=20, num_layers=3, 
                 seed=42, num_bases=None):
    set_seed(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    node_type = 'entity' if 'entity' in data.node_types else data.node_types[0]
    x = data[node_type].x.to(device) 
    num_nodes = x.size(0)
    x_truth_semantic = F.normalize(x.detach().clone(), p=2, dim=1) 
    edge_index, edge_type, rels = build_edge_index_and_types(data, device)
    num_relations = len(rels)
    
    in_channels = x.size(1)
    model = RGCNEncoder(in_channels, hidden, in_channels, num_relations, num_layers, dropout=0.2, num_bases=num_bases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience)
    mse_loss_fn = nn.MSELoss() 
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2.0) 

    best_loss = float('inf')
    epochs_no_improve = 0 
    
    if edge_index.size(1) > 0:
        pos_anchors, pos_partners = edge_index[0], edge_index[1]
    else:
        pos_anchors = pos_partners = torch.tensor([], dtype=torch.long, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        h = model(x, edge_index, edge_type)
        loss_anchor = mse_loss_fn(h, x_truth_semantic)
        if pos_anchors.numel() > 0:
            neg_indices = torch.randint(0, num_nodes, (pos_anchors.size(0),), device=device)
            mask = (neg_indices == pos_partners)
            neg_indices[mask] = (neg_indices[mask] + 1) % num_nodes
            loss_explore = triplet_loss_fn(h[pos_anchors], h[pos_partners], h[neg_indices])
        else:
            loss_explore = torch.tensor(0.0, device=device)
        
        loss = (alpha * mse_scale * loss_anchor) + (beta * loss_explore)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_no_improve = 0 
            torch.save({'embeddings': h.detach().cpu(), 'rel_names': rels}, output_path + ".best.pt")
        else:
            epochs_no_improve += 1 

        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            break 
            
    if os.path.exists(output_path + ".best.pt"):
        out_dict = torch.load(output_path + ".best.pt", map_location='cpu')
        torch.save(out_dict, output_path)
        os.remove(output_path + ".best.pt") 
        return out_dict['embeddings']
    return None


def train_hybrid_gcn(data, hidden, lr, epochs, device, output_path,
                     alpha=0.8, beta=0.2, margin=1.0, mse_scale=50.0,
                     weight_decay=1e-5, scheduler_patience=20,
                     early_stopping_patience=20, num_layers=3, seed=42):
    set_seed(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = x.size(0)
    x_truth_semantic = F.normalize(x.detach().clone(), p=2, dim=1)

    model = GCNEncoder(
        in_channels=x.size(1),
        hidden_channels=hidden,
        out_channels=x.size(1),
        num_layers=num_layers,
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )
    mse_loss_fn = nn.MSELoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2.0)

    best_loss = float("inf")
    epochs_no_improve = 0
    if edge_index.size(1) > 0:
        pos_anchors = edge_index[0]
        pos_partners = edge_index[1]
    else:
        pos_anchors = pos_partners = torch.tensor([], dtype=torch.long, device=device)

    for _epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        h = model(x, edge_index)
        loss_anchor = mse_loss_fn(h, x_truth_semantic)
        if pos_anchors.numel() > 0:
            neg_indices = torch.randint(0, num_nodes, (pos_anchors.size(0),), device=device)
            mask = neg_indices == pos_partners
            neg_indices[mask] = (neg_indices[mask] + 1) % num_nodes
            loss_explore = triplet_loss_fn(h[pos_anchors], h[pos_partners], h[neg_indices])
        else:
            loss_explore = torch.tensor(0.0, device=device)

        loss = (alpha * mse_scale * loss_anchor) + (beta * loss_explore)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_no_improve = 0
            torch.save({"embeddings": h.detach().cpu()}, output_path + ".best.pt")
        else:
            epochs_no_improve += 1

        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            break

    if os.path.exists(output_path + ".best.pt"):
        out_dict = torch.load(output_path + ".best.pt", map_location="cpu")
        torch.save(out_dict, output_path)
        os.remove(output_path + ".best.pt")
        return out_dict["embeddings"]
    return None


def split_edges(num_edges, train_ratio=0.85, val_ratio=0.05, device=None):
    if device is None:
        device = torch.device("cpu")
    indices = torch.randperm(num_edges, device=device)
    n_train = int(num_edges * train_ratio)
    n_val = int(num_edges * val_ratio)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


@torch.no_grad()
def evaluate_transe_mrr(model, x, edge_index, edge_type, eval_idx, batch_size=100):
    model.eval()
    if eval_idx.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0

    all_emb = model(x, edge_index, edge_type)
    rel_emb = model.relation_embedding
    src = edge_index[0, eval_idx]
    dst = edge_index[1, eval_idx]
    rel = edge_type[eval_idx]
    ranks = []

    for i in range(0, eval_idx.numel(), batch_size):
        b_src = src[i:i + batch_size]
        b_dst = dst[i:i + batch_size]
        b_rel = rel[i:i + batch_size]
        query = all_emb[b_src] + rel_emb[b_rel]
        dists = torch.cdist(query, all_emb, p=2)
        true_dists = dists.gather(1, b_dst.view(-1, 1))
        rank = (dists < true_dists).sum(dim=1).float() + 1
        ranks.append(rank)

    ranks = torch.cat(ranks)
    mrr = (1.0 / ranks).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    return mrr, hits1, hits3, hits10


def train_multi_edge_rgcn_transe(
    data,
    hidden,
    lr,
    epochs,
    device,
    output_path,
    alpha=0.8,
    beta=0.2,
    margin=1.0,
    mse_scale=50.0,
    weight_decay=1e-5,
    scheduler_patience=20,
    early_stopping_patience=30,
    num_layers=3,
    seed=42,
    num_bases=None,
    eval_interval=10,
):
    set_seed(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    node_type = "entity" if "entity" in data.node_types else data.node_types[0]
    x = data[node_type].x.to(device)
    x_truth_semantic = F.normalize(x.detach().clone(), p=2, dim=1)
    edge_index, edge_type, rels = build_edge_index_and_types(data, device)
    if edge_index.size(1) == 0:
        raise ValueError("Multi-edge R-GCN TransE training requires at least one edge.")

    num_nodes = x.size(0)
    num_relations = len(rels)
    train_idx, val_idx, _test_idx = split_edges(edge_index.size(1), device=device)

    model = RGCNTransE(
        in_channels=x.size(1),
        hidden_channels=hidden,
        out_channels=x.size(1),
        num_relations=num_relations,
        num_layers=num_layers,
        dropout=0.2,
        num_bases=num_bases,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.8, patience=scheduler_patience
    )
    mse_loss_fn = nn.MSELoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2.0)

    best_metric = -float("inf")
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        h = model(x, edge_index, edge_type)
        loss_anchor = mse_loss_fn(h, x_truth_semantic)

        if train_idx.numel() > 0:
            batch_size = min(train_idx.numel(), 10000)
            perm = torch.randperm(train_idx.numel(), device=device)[:batch_size]
            batch_idx = train_idx[perm]
            src = edge_index[0, batch_idx]
            dst = edge_index[1, batch_idx]
            rel = edge_type[batch_idx]
            neg_dst = torch.randint(0, num_nodes, (batch_size,), device=device)
            anchor_term = h[src] + model.relation_embedding[rel]
            loss_link = triplet_loss_fn(anchor_term, h[dst], h[neg_dst])
        else:
            loss_link = torch.tensor(0.0, device=device)

        loss = (alpha * mse_scale * loss_anchor) + (beta * loss_link)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        should_eval = epoch % max(1, eval_interval) == 0 or epoch == epochs
        improved = False
        if should_eval and val_idx.numel() > 0:
            val_mrr, _h1, _h3, _h10 = evaluate_transe_mrr(
                model, x, edge_index, edge_type, val_idx
            )
            scheduler.step(val_mrr)
            if val_mrr > best_metric:
                best_metric = val_mrr
                improved = True
        elif loss.item() < best_loss:
            best_loss = loss.item()
            improved = True

        if improved:
            epochs_no_improve = 0
            torch.save(
                {
                    "embeddings": h.detach().cpu(),
                    "rel_embeddings": model.relation_embedding.detach().cpu(),
                    "rel_names": rels,
                    "mrr": best_metric if best_metric > -float("inf") else None,
                },
                output_path + ".best.pt",
            )
        elif should_eval:
            epochs_no_improve += max(1, eval_interval)

        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            break

    if os.path.exists(output_path + ".best.pt"):
        out_dict = torch.load(output_path + ".best.pt", map_location="cpu")
        torch.save(out_dict, output_path)
        os.remove(output_path + ".best.pt")
        return out_dict["embeddings"]
    return None
