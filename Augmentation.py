import torch
import math

from torch_geometric.utils import degree, to_undirected
from time import time

# ——————————————————————————————Edge Processing——————————————————————————————
def add_edges(edge_weights, edge_index, former_index, p = 0.5, threshold = 0.9):
    # p: edge adding rate
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
    add_index = edge_index[:, sel_mask]
    return torch.cat((former_index, add_index), dim=1)

def add_edge_weights(edge_index, p):
    device = edge_index.device
    edge_index = to_undirected(edge_index)
    # degrees of nodes
    deg = degree(edge_index[1])
    # maximum size of edges to be added
    l = math.ceil(edge_index.size(1) * p)
    # edges index that can be added
    add_index = reverse(edge_index).to(device)

    # calculate weights
    w = 1 / ((deg[add_index[0]] + 1)*(deg[add_index[1]] + 1)) ** 0.5
    weights = (w.max() - w) / (w.max() - w.mean())
    # keep the top l only  ## to accelerate
    sel_mask = torch.topk(weights, l)[1]
    edge_weights = weights[sel_mask]
    add_edge = add_index[:, sel_mask]
    return edge_weights, add_edge

def add_edge_random(edge_index, p=0.1):
    device = edge_index.device
    index = to_undirected(edge_index)
    l = index.size(1)
    add_index = torch.tensor(reverse(index)).to(device)
    l2 = add_index.size(1)
    p = p * l / l2
    weights = (torch.ones(l2) * p).to(device)
    sel_mask = torch.bernoulli(weights).to(torch.bool)
    add_index = add_index[:, sel_mask]
    return torch.cat((edge_index, add_index), dim=1)

def drop_edges(edge_index, edge_weights, p, threshold=1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]

def drop_edge_weights(edge_index):
    edge_index = to_undirected(edge_index)
    deg = degree(edge_index[1])
    w = 1 / (deg[edge_index[0]] * deg[edge_index[1]]) ** 0.5
    weights = (w - w.min()) / (w.mean() - w.min())
    return weights

# ——————————————————————————————Feature Processing——————————————————————————————

def drop_feature_random(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

# ——————————————————————————————Auxiliary Function——————————————————————————————

def get_index(node_num, node_list):
    return (node_list == node_num).nonzero()

def get_edge_index(target, edge_index):
    edge_index = to_undirected(edge_index)
    deg = degree(edge_index[0]).to(torch.int32)
    output = torch.tensor([]).to(edge_index.device)
    for t in target:
        m = deg[:t].sum()
        d = deg[t]
        output = torch.cat((output, edge_index[:, m:m + d]), dim = -1)
    return output

def reverse(edge_index):
    edge_index = to_undirected(edge_index)
    node_num = edge_index[0].max() + 1
    if node_num > 30000:
        print("Too many nodes, processing")
        t1 = time()
        # select nodes
        deg = degree(edge_index[1])
        len = edge_index.size(1)
        candidate_num = min(node_num, math.ceil(math.sqrt(2 * len)))
        candidate_nodes = torch.topk(deg, candidate_num)[1]
        
        # get edge index
        edge_index = get_edge_index(candidate_nodes, edge_index)
        mask = torch.tensor([k in candidate_nodes for k in edge_index[1]])
        edge_index = edge_index[:, mask]

        # candidate_nodes → [0, candidate_num - 1]
        edge_index_1 = torch.tensor([get_index(u, candidate_nodes) for u in edge_index[0]]).view(1, -1)
        edge_index_2 = torch.tensor([get_index(u, candidate_nodes) for u in edge_index[1]]).view(1, -1)
        edge_index_ = torch.cat((edge_index_1, edge_index_2))
        device = edge_index_.device
        
        # reverse & [0, candidate_num - 1] → candidate_nodes
        aux = torch.ones(edge_index_.shape[1]).to(device)
        reverse_index_ = (1 - torch.sparse_coo_tensor(edge_index_, aux).to_dense()).to_sparse().indices()
        reverse_index = candidate_nodes[reverse_index_]
        t2 = time()
        print(f"Completed, processing time:{t2-t1:.4f}")
        return reverse_index
    else:
        device = edge_index.device
        aux = torch.ones(edge_index.shape[1]).to(device)
        return (1 - torch.sparse_coo_tensor(edge_index, aux).to_dense()).to_sparse().indices()