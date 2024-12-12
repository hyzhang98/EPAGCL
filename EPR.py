import torch
import numpy as np

from torch_geometric.utils import degree, to_undirected

from Load_data import get_dataset

data = get_dataset('../datasets/Cora', 'Cora')
label = data.y
edge_index = data.edge_index
edge_index = to_undirected(edge_index)
deg = degree(edge_index[0]).to(torch.int32)
computed_deg = 0
epr = []
for i in range(label.size(0)):
    d = deg[i]
    if d == 0:
        continue
    neighbour = edge_index[1][computed_deg : computed_deg + d]
    computed_deg += d
    neighbour_deg = deg[neighbour]
    neighbour_y = label[neighbour]
    wrong_neighbour_mask = (neighbour_y != label[i])
    wrong_passing = 1 / neighbour_deg[wrong_neighbour_mask] ** 0.5
    all_passing = 1 / neighbour_deg ** 0.5
    epr.append((wrong_passing.sum()/all_passing.sum()).item())
epr = np.array(epr)
print(epr.mean())
# epr of each node, average of which usually under 0.3