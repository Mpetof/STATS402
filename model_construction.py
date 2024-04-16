import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,MessagePassing
from torch_geometric.utils import add_self_loops
import gzip
import pickle

# Model Constructions

#################################GCNPair#################################
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

#################################GATPair#################################
class GATPair(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super().__init__()
        self.conv1_l = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        self.conv2_l = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv3_l = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv1_r = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        self.conv2_r = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv3_r = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_l, graph_r, labels):
        x_l = self.conv1_l(graph_l.x, graph_l.edge_index).relu()
        x_l = F.dropout(x_l, p=0.5, training=self.training)
        x_l = self.conv2_l(x_l, graph_l.edge_index).relu()
        x_l = F.dropout(x_l, p=0.15, training=self.training)
        x_l = self.conv3_l(x_l, graph_l.edge_index).relu()
        
        x_r = self.conv1_r(graph_r.x, graph_r.edge_index).relu()
        x_r = F.dropout(x_r, p=0.5, training=self.training)
        x_r = self.conv2_r(x_r, graph_r.edge_index).relu()
        x_r = F.dropout(x_r, p=0.15, training=self.training)
        x_r = self.conv3_l(x_r, graph_r.edge_index).relu()
        
        x_merge = merge_graphs(x_l, x_r, labels)
        x = F.relu(self.fc1(x_merge))
        out = self.fc2(x)
        return out

#################################GCNPair_EdegEm#################################
class CustomGCNConv_1(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels):
        # 'add' as the aggregator
        super().__init__(aggr='add')
        self.lin_self = torch.nn.Linear(in_channels, out_channels)
        self.lin_neigh = torch.nn.Linear(in_channels, out_channels)
        self.lin_edge = torch.nn.Linear(edge_channels,1)

        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to the edge_index, basically a self message passing
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0))

        # Transform self features
        self_feature = self.lin_self(x)

        # Start message passing
        return self_feature + self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_index_i, size_i, edge_attr):
        # x_j: target nodes
        # size_i: number of neighbors, the normalizing constant
        # edge_attr: edge feature
        neigh_feature = self.lin_neigh(x_j)
        #Make sure edge_feature has the same shape as neigh_feature!!!
        edge_feature = self.lin_edge(edge_attr).squeeze(-1)  
        edge_feature = edge_feature.unsqueeze(-1) if edge_feature.dim() == 1 else edge_feature
        return neigh_feature + edge_feature

    def update(self, aggr_out):
        # aggr_out: The feature after self.propagate
        return aggr_out + self.bias
        

# Model Construction
class GCNPair_EdegEm(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim):
        super().__init__()
        torch.manual_seed(8)
        self.conv1_l = CustomGCNConv_1(input_dim, hidden_dim, edge_dim)
        self.conv2_l = CustomGCNConv_1(hidden_dim, hidden_dim, edge_dim)
        self.conv1_r = CustomGCNConv_1(input_dim, hidden_dim, edge_dim)
        self.conv2_r = CustomGCNConv_1(hidden_dim, hidden_dim, edge_dim)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_l, graph_r,labels):
        x_l = self.conv1_l(graph_l.x, graph_l.edge_index, graph_l.edge_attr).relu()
        x_l = F.dropout(x_l, p=0.5, training=self.training)
        x_l = self.conv2_l(x_l, graph_l.edge_index, graph_l.edge_attr)

        x_r = self.conv1_r(graph_r.x, graph_r.edge_index, graph_r.edge_attr).relu()
        x_r = F.dropout(x_r, p=0.5, training=self.training)
        x_r = self.conv2_r(x_r, graph_r.edge_index, graph_r.edge_attr)

        # Assuming merge_graphs function is defined elsewhere in your code
        # and 'labels' is an argument provided where forward is called
        x_merge = merge_graphs(x_l, x_r,labels)
        x = F.relu(self.fc1(x_merge))
        out = self.fc2(x)
        return out


##################################################################

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
