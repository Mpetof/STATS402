import torch
from model_construction import GCNPair, GATPair, GCNPair_EdegEm
from model_construction import map_labels, load_cpkl_gz, load_data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Define the same dictionary of models for consistency
models_dict = {
    'gcn_pair': {
        'class': GCNPair,
        'params': {
            'input_dim': 70,
            'hidden_dim': 32,
            'output_dim': 2
        }
    },
    'gat_pair': {
        'class': GATPair,
        'params': {
            'input_dim': 70,
            'hidden_dim': 32,
            'output_dim': 2
        }
    },
    'gcn_pair_edge': {
        'class': GCNPair_EdegEm,
        'params': {
            'input_dim': 70,
            'hidden_dim': 45,
            'output_dim': 2,
            'edge_dim': 2
        }
    }
}

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f"./results/{model_name}_confusion_matrix.pdf")
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"./results/{model_name}_roc_curve.pdf")
    plt.show()

# Inference function
def inference(model, test_data, model_name, device):
    model.eval()
    y_true, y_scores, y_pred = [], [], []
    
    with torch.no_grad():
        for pair_id in range(len(test_data[1])):
            graph_l, graph_r, label = load_data('test', pair_id, test_data, device)
            label_mapped = map_labels(label[:,-1])
            label_tensor = torch.tensor(label_mapped, dtype=torch.long).to(device)
            
            out = model(graph_l, graph_r,label)
            prob = F.softmax(out, dim=1)[:, 1]
            y_true.extend(label_tensor.cpu().numpy())
            y_scores.extend(prob.cpu().numpy())
            y_pred.extend((prob > 0.5).cpu().numpy().astype(int))
    
    auc_score = roc_auc_score(y_true, y_scores)
    print(f'Test AUC for {model_name}: {auc_score:.4f}')
    
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc_curve(y_true, y_scores, model_name)

# Main inference routine
if __name__ == "__main__":
    model_choice = input("Enter the model choice (from the list above): ")
    model_path = input("Enter the path to the model parameters file (e.g., './models/GCN_node_model.pth'): ")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")
    
    if model_choice in models_dict:
        model_class = models_dict[model_choice]['class']
        model_params = models_dict[model_choice]['params']
        model = model_class(**model_params).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        test_data = load_cpkl_gz('./dataset/test.cpkl.gz')
        inference(model, test_data, model_choice, device)
    else:
        print(f"Error: '{model_choice}' is not a valid model choice.")
