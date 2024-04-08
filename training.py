import torch
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc,roc_auc_score
import seaborn as sns
from model_construction import GCNPair,GATPair
from model_construction import map_labels, load_cpkl_gz, load_data
import torch.nn.functional as F

###################################BASIC_SETTINGS_START##########################################
# Define a dictionary of models
models_dict = {
    'gcn_pair': {
        'class': GCNPair,
        'params': {
            'input_dim': 70,
            'hidden_dim': 32,  # This will be taken from user input
            'output_dim': 2
        }
    },
    'gat_pair': {
        'class': GATPair,
        'params': {
            'input_dim': 70,
            'hidden_dim': 32,  # This will be taken from user input
            'output_dim': 2
        }
    },
    # Add other models to the dictionary in the same manner
    # 'other_model': {
    #     'class': OtherModelClass,
    #     'params': {
    #         'input_dim': 70,
    #         # Other parameters specific to this model
    #     }
    # },
}

# Print available model choices
print("Available models:")
for model_name in models_dict:
    print(f"- {model_name}")

model_choice = input("Enter the model choice (from the list above): ")

# Ensure that the model choice is valid
if model_choice not in models_dict:
    print(f"Error: '{model_choice}' is not a valid model choice.")
    assert model_choice in models_dict, f"Error: '{model_choice}' is not a valid model choice."

# Request user input for configurable parameters
epochs = int(input("Enter the number of epochs: "))
learning_rate = float(input("Enter the learning rate: "))
model_save_name = input("Enter the model save name (e.g., GCN_node_model.pth): ")

# Device Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

###################################BASIC_SETTINGS_END############################################


# Train and Test functions
def train_one_epoch(model, optimizer, criterion, train_data, epoch):
    model.train()
    total_loss = 0
    y_true = [] 
    y_scores = [] 

    for pair_id in range(len(train_data[1])):
        graph_l, graph_r, label = load_data('train', pair_id, train_data,device)
        label_mapped = map_labels(label[:,-1])  # CE loss expects 0/1 labels for y
        label_tensor = torch.tensor(label_mapped, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        out = model(graph_l, graph_r,label) 
        loss = criterion(out, label_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        prob = F.softmax(out, dim=1)[:, 1].detach()  
        y_true.extend(label_tensor.cpu().numpy())
        y_scores.extend(prob.cpu().numpy())

    avg_loss = total_loss / len(train_data[1])
    auc_score = roc_auc_score(y_true, y_scores) 

    print(f'Epoch: {epoch:03d}, Loss: {avg_loss:.5f}, Training AUC: {auc_score:.5f}')
    return avg_loss, auc_score

def test(model, criterion, test_data, threshold=0.5):
    model.eval()
    y_true = []
    y_scores = []
    y_pred = []
    with torch.no_grad():
        for pair_id in range(len(test_data[1])):
            graph_l, graph_r, label = load_data('test', pair_id, test_data,device)
            label_mapped = map_labels(label[:,-1])
            label_tensor = torch.tensor(label_mapped, dtype=torch.long).to(device)
            out = model(graph_l, graph_r, label)
            prob = F.softmax(out, dim=1)[:, 1]
            y_true.extend(label_tensor.cpu().numpy())
            y_scores.extend(prob.cpu().numpy())
            y_pred.extend((prob >= threshold).cpu().numpy().astype(int))
    auc_score = roc_auc_score(y_true, y_scores)
    return auc_score, y_true, y_scores, y_pred

def plot_training_graphs(epochs, train_losses, train_aucs, model_name):
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, train_aucs, label='Train AUC', color='orange')

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Train Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./results/{model_name}_training_graphs.pdf")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f"./results/{model_name}_confusion_matrix.pdf")
    plt.show()


def plot_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def main():
    train_data = load_cpkl_gz('./dataset/train.cpkl.gz')
    test_data = load_cpkl_gz('./dataset/test.cpkl.gz')
    
    # Add a patience parameter for early stopping
    patience = 7
    best_auc = 0  # Initialize best AUC
    epochs_no_improve = 0  # Initialize counter for early stopping
    last_epoch = 0 

    # Instantiate the selected model with its parameters
    model_class = models_dict[model_choice]['class']
    model_params = models_dict[model_choice]['params']
    model = model_class(**model_params).to(device)

    #model = GCNPair(input_dim=70, hidden_dim=hidden_dim, output_dim=2).to(device) #No longer Used
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weight = torch.tensor([1, 6], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    train_losses, train_aucs = [], []
    start_time = time.time()

    #Early Stopping is added
    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_one_epoch(model, optimizer, criterion, train_data, epoch)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        last_epoch = epoch
        
        # Early stopping logic
        if train_auc > best_auc:
            best_auc = train_auc
            epochs_no_improve = 0
            # Save the best model if AUC improved
            torch.save(model.state_dict(), f'./models/{model_save_name}_best.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch} epochs!')
                break  # Early stoppinp
    
    end_training_time = time.time()


    test_auc, y_true, y_scores,y_pred = test(model, criterion, test_data)
    print(f'Test AUC: {test_auc:.4f}')
    end_testing_time = time.time()

    print(f'Test AUC: {test_auc:.4f}')
    print(f"Training took {end_training_time - start_time:.2f} seconds.")
    print(f"Inference took {end_testing_time - end_training_time:.2f} seconds.")

    torch.save(model.state_dict(), f'./models/{model_save_name}')

    model_base_name = model_save_name.replace('.pth', '') 
    plot_training_graphs(range(1, last_epoch + 1), train_losses, train_aucs, model_base_name)
    plot_confusion_matrix(y_true, y_pred, model_base_name)
    plot_roc_curve(y_true, y_pred,model_base_name)

if __name__ == "__main__":
    main()
