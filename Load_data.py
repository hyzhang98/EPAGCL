from torch_geometric.datasets import WikiCS, Coauthor, Amazon, Planetoid, WebKB
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
from torch_geometric import transforms as T

def get_dataset(path, name):
    assert name in ['WikiCS', 'Coauthor-CS', 'Coauthor-Phy', 'Amazon-Computers', 'Amazon-Photo', 'Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv', 'texas', 'cornell', 'wisconsin']

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())[0]

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())[0]

    if name == 'WikiCS':
        return WikiCS(root=path)[0]

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())[0]

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())[0]
    
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(root=path, name=name)[0]

    if name == 'ogbn-arxiv':
        return PygNodePropPredDataset(root=path, name=name, transform=T.ToUndirected())[0]
    
    if name in ['texas', 'cornell', 'wisconsin']:
        data = WebKB(root=path, name=name, transform=T.NormalizeFeatures())[0]
        data.edge_index = to_undirected(data.edge_index)
        return data