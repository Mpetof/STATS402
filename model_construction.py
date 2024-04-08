import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
import gzip
import pickle

# Model Constructions

class GCNPair(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1_l = GCNConv(input_dim, hidden_dim, aggr='add')
        self.conv2_l = GCNConv(hidden_dim, hidden_dim, aggr='add')
        self.conv1_r = GCNConv(input_dim, hidden_dim, aggr='add')
        self.conv2_r = GCNConv(hidden_dim, hidden_dim, aggr='add')
        self.fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_l, graph_r, labels):
        x_l = self.conv1_l(graph_l.x, graph_l.edge_index).relu()
        x_l = F.dropout(x_l, p=0.6, training=self.training)
        x_l = self.conv2_l(x_l, graph_l.edge_index)

        x_r = self.conv1_r(graph_r.x, graph_r.edge_index).relu()
        x_r = F.dropout(x_r, p=0.6, training=self.training)
        x_r = self.conv2_r(x_r, graph_r.edge_index)

        x_merge = merge_graphs(x_l, x_r, labels)
        x = F.relu(self.fc1(x_merge))
        out = self.fc2(x)
        return out

class GATPair(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super().__init__()
        self.conv1_l = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        self.conv2_l = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv1_r = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        self.conv2_r = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_l, graph_r, labels):
        x_l = self.conv1_l(graph_l.x, graph_l.edge_index).relu()
        x_l = F.dropout(x_l, p=0.6, training=self.training)
        x_l = self.conv2_l(x_l, graph_l.edge_index).relu()

        x_r = self.conv1_r(graph_r.x, graph_r.edge_index).relu()
        x_r = F.dropout(x_r, p=0.6, training=self.training)
        x_r = self.conv2_r(x_r, graph_r.edge_index).relu()

        x_merge = merge_graphs(x_l, x_r, labels)
        x = F.relu(self.fc1(x_merge))
        out = self.fc2(x)
        return out

# Some important functions
def merge_graphs(x_l, x_r, labels):
    merged_features = []
    for label in labels:
        l_index, r_index, _ = label
        l_features = x_l[l_index]
        r_features = x_r[r_index]
        merged_feature = torch.cat((l_features, r_features), dim=0)
        merged_features.append(merged_feature)
    merged_features = torch.stack(merged_features)
    return merged_features

def map_labels(label):
    return (label + 1) // 2

def load_cpkl_gz(filename):
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def load_data(folder, pair_id, data, device):
    # Load graph data objects
    with open(f'./constructed_graphs/{folder}/protein_pair_{pair_id}_l.pkl', 'rb') as f:
        graph_l = pickle.load(f).to(device)
    with open(f'./constructed_graphs/{folder}/protein_pair_{pair_id}_r.pkl', 'rb') as f:
        graph_r = pickle.load(f).to(device)
    label = data[1][pair_id]['label']
    return graph_l, graph_r, label
