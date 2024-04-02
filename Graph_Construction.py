"""
How to Use This Script:
1. Ensure you have Python installed on your system.
2. Save this script to a file, for example, 'process_protein_interactions.py'.
3. Run the script with either the default paths or specify custom paths using command-line arguments:
   - To run with default paths:
     python process_protein_interactions.py
   - To specify custom paths:
     python process_protein_interactions.py --train_path <custom_train_path> --test_path <custom_test_path> --graph_folder_path_train <custom_graph_folder_path_train> --graph_folder_path_test <custom_graph_folder_path_test>
   Replace <custom_train_path>, <custom_test_path>, <custom_graph_folder_path_train>, and <custom_graph_folder_path_test> with your actual file and directory paths.
"""

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import gzip
import pickle
import os
import argparse

def setup_device():
    try:
        device = torch.device('cuda')
        print("CUDA Available, Running on GPU")
    except:
        print("No CUDA Available, Running on CPU")

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_cpkl_gz(filename):
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def convert_to_edges_undirected(data, protein_loc, side='l'):
    hood_indices_key = f'{side}_hood_indices'
    l_hood_indices = data[1][protein_loc][hood_indices_key]
    n_residues = l_hood_indices.shape[0]
    adjacency_matrix = np.zeros((n_residues, n_residues), dtype=int)
    for residue_index in range(n_residues):
        neighbors = l_hood_indices[residue_index, :, 0]
        adjacency_matrix[residue_index, neighbors] = 1
        adjacency_matrix[neighbors, residue_index] = 1
    assert np.all(adjacency_matrix == adjacency_matrix.T), "Adjacency matrix must be symmetric."
    edges_array = np.array(np.triu(adjacency_matrix).nonzero())
    return edges_array

def construct_edge_features(edge_array, data, protein_loc, side='l'):
    edge_dict = {}
    hood_indices_key = f'{side}_hood_indices'
    edge_key = f'{side}_edge'
    for i in range(edge_array.shape[1]):
        key = tuple(edge_array[:, i])
        parent_residue, child_residue = key
        children_search_range = data[1][protein_loc][hood_indices_key][parent_residue][:, 0]
        search_result = np.where(children_search_range == child_residue)
        if len(search_result[0]) == 0:
            parent_residue, child_residue = child_residue, parent_residue
            children_search_range = data[1][protein_loc][hood_indices_key][parent_residue][:, 0]
            search_result = np.where(children_search_range == child_residue)
        if len(search_result[0]) > 0:
            index = search_result[0][0]
            value = data[1][protein_loc][edge_key][parent_residue][index]
            edge_dict[key] = value
        else:
            print(f"No match found for {key}")
    edge_features = np.array(list(edge_dict.values()))
    return edge_features

def construct_labels(data, protein_loc):
    labels_detail = sorted(data[1][protein_loc]['label'].tolist(), key=lambda x: (x[0], x[1]))
    labels = [sublist[2] for sublist in labels_detail]
    return labels

def construct_nodes(data, protein_loc, side='l'):
    vertex_key = f'{side}_vertex'
    return data[1][protein_loc][vertex_key]

def construct_and_save_graphs(data, protein_loc, graph_folder_path, side):
    edge_array = convert_to_edges_undirected(data, protein_loc, side)
    edge_features = construct_edge_features(edge_array, data, protein_loc, side)
    labels = construct_labels(data, protein_loc)
    nodes = construct_nodes(data, protein_loc, side)
    
    graph_data = Data(x=torch.tensor(nodes, dtype=torch.float), 
                      edge_index=torch.tensor(edge_array, dtype=torch.long), 
                      edge_attr=torch.tensor(edge_features, dtype=torch.float), 
                      y=torch.tensor(labels, dtype=torch.long))
    
    file_name = f"protein_pair_{protein_loc}_{side}.pkl"
    file_path = os.path.join(graph_folder_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(graph_data, f)

def main(train_path, test_path, graph_folder_path_train, graph_folder_path_test):
    setup_device()
    os.environ['TORCH'] = torch.__version__
    print(torch.__version__)
    ensure_directory_exists(graph_folder_path_train)
    ensure_directory_exists(graph_folder_path_test)
    train_data = load_cpkl_gz(train_path)
    test_data = load_cpkl_gz(test_path)
    print("Following elements for one observation: "),
    print(list(train_data[1][0].keys()))

    # Loop over train_data and test_data to construct and save graphs
    for protein_loc in range(len(train_data[1])):
        construct_and_save_graphs(train_data, protein_loc, graph_folder_path_train, 'l')  # Left side
        construct_and_save_graphs(train_data, protein_loc, graph_folder_path_train, 'r')  # Right side
    for protein_loc in range(len(test_data[1])):
        construct_and_save_graphs(test_data, protein_loc, graph_folder_path_test, 'l')  # Left side
        construct_and_save_graphs(test_data, protein_loc, graph_folder_path_test, 'r')  # Right side

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process protein interaction graphs.")
    parser.add_argument('--train_path', type=str, default='./dataset/train.cpkl.gz', help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, default='./dataset/test.cpkl.gz', help='Path to the testing dataset')
    parser.add_argument('--graph_folder_path_train', type=str, default='./constructed_graphs/train', help='Path to save constructed graphs for training data')
    parser.add_argument('--graph_folder_path_test', type=str, default='./constructed_graphs/test', help='Path to save constructed graphs for testing data')

    args = parser.parse_args()
    main(args.train_path, args.test_path, args.graph_folder_path_train, args.graph_folder_path_test)
