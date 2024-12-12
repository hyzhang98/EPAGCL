import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_proj_hidden, temperature=0.3, activation='PReLU'):
        super().__init__()
        if activation == 'PReLU':
            self.activation = nn.PReLU()
        self._build_up(in_channels, out_channels, num_proj_hidden, temperature)

    def forward(self, x, edge_index):
        out = self.activation(self.conv1(x, edge_index))
        out = self.activation(self.conv2(out, edge_index))
        proj = F.relu(self.fc1(out))
        proj = self.fc2(proj)
        return out, proj
    
    def loss(self, z1, z2, batch_compute= False, batch_size=256):
        l1 = self.cal_loss(z1, z2, batch_compute, batch_size)
        l2 = self.cal_loss(z2, z1, batch_compute, batch_size)
        l = (l1 + l2) * 0.5
        return l.mean()
    
    def cal_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim = torch.mm(z1, z2.t())
        return torch.exp(sim / self.t)

    def cal_loss(self, z1, z2, batch_compute=False, batch_size=256):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        if not batch_compute:
            indices = torch.arange(0, num_nodes).to(device)
        else:
            indices = torch.randperm(num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            if not batch_compute:
                refl_sim = self.cal_sim(z1[mask], z1)
                between_sim = self.cal_sim(z1[mask], z2)
                losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
            else:
                refl_sim = self.cal_sim(z1[mask], z1[mask])
                between_sim = self.cal_sim(z1[mask], z2[mask])
                losses.append(-torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())))
        return torch.cat(losses)    

    def _build_up(self, in_channels, out_channels, num_proj_hidden, temperature):
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.fc1 = nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, out_channels)
        self.t = temperature
    
    def freeze(self, flag=True):
        for p in self.parameters():
            p.requires_grad = not flag


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, classes_num):
        super().__init__()
        self.fc = nn.Linear(feat_dim, classes_num)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        out = self.fc(x)
        return out
