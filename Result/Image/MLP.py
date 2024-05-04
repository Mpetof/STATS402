import os
import gzip
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def load_cpkl_gz(filename):
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

class ProteinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = np.array(features)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_dataset = ProteinDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ProteinDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer4 = nn.Linear(hidden_size // 4, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.tanh(self.layer1(x))
        out = self.tanh(self.layer2(out))
        out = self.tanh(self.layer3(out))
        out = self.layer4(out)
        return out.view(-1)

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
    plt.savefig(f"./results/Experiments_MLP/{model_name}_training_graphs_3.pdf")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f"./results/Experiments_MLP/{model_name}_confusion_matrix_3.pdf")
    plt.show()

def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"./results/Experiments_MLP/{model_name}_roc_curve_3.pdf")
    plt.show()

def train(model, train_loader, criterion, optimizer, num_epochs, patience):
    model.train()
    best_auc = 0
    epochs_no_improve = 0
    early_stop = False
    train_losses = []
    train_aucs = []
    epochs_list = []

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping")
            break

        total_loss = 0
        targets_all, outputs_all = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            outputs_all.extend(outputs.sigmoid().detach().cpu().numpy())
            targets_all.extend(labels.detach().cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_auc = roc_auc_score(targets_all, outputs_all)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        epochs_list.append(epoch + 1)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')

        # Save the model if the AUC has improved
        if train_auc > best_auc:
            best_auc = train_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), './models/MLPbest_model_3.pth')
            print("Model improved and saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
                model.load_state_dict(torch.load('./models/MLPbest_model_3.pth'))

    plot_training_graphs(epochs_list, train_losses, train_aucs, 'MLP')

def test(model, test_loader):
    model.eval()
    targets_all, outputs_all = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            outputs_all.extend(outputs.sigmoid().detach().cpu().numpy())
            targets_all.extend(labels.detach().cpu().numpy())
    test_auc = roc_auc_score(targets_all, outputs_all)
    print(f'Test AUC: {test_auc:.4f}')

    # Plot results
    plot_confusion_matrix(targets_all, (np.array(outputs_all) > 0.5).astype(int), 'MLP')
    plot_roc_curve(targets_all, outputs_all, 'MLP')

    return targets_all, outputs_all

def main():
    train_data = load_cpkl_gz('./dataset/train.cpkl.gz')
    test_data = load_cpkl_gz('./dataset/test.cpkl.gz')
    X_train, y_train, X_test, y_test = [], [], [], []

    for protein in train_data[1]:
        for pair in range(len(protein['label'])):
            ind = protein['label'][pair]
            X = np.concatenate([protein['l_vertex'][ind[0]].flatten(),
                                protein['l_edge'][ind[0]].flatten(),
                                protein['r_vertex'][ind[1]].flatten(),
                                protein['r_edge'][ind[1]].flatten()])
            y = (ind[2] + 1) // 2
            X_train.append(X)
            y_train.append(y)

    for protein in test_data[1]:
        for pair in range(len(protein['label'])):
            ind = protein['label'][pair]
            X = np.concatenate([protein['l_vertex'][ind[0]].flatten(),
                                protein['l_edge'][ind[0]].flatten(),
                                protein['r_vertex'][ind[1]].flatten(),
                                protein['r_edge'][ind[1]].flatten()])
            y = (ind[2] + 1) // 2
            X_test.append(X)
            y_test.append(y)

    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    print("Data Loading Finished!")
    input_size = 220
    hidden_size = 480
    model = MLP(input_size, hidden_size).to(device)
    pos_weight = torch.tensor([6], dtype=torch.float).to(device)#Note set this weight to 0.2 is good
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)

    start_time = time.time()
    train(model, train_loader, criterion, optimizer, 100, 10)
    end_time = time.time()
    print(f"Training finished in {end_time - start_time} seconds.")

    start_time = time.time()
    targets, scores = test(model, test_loader)
    end_time = time.time()
    print(f"Testing finished in {end_time - start_time} seconds.")
    
if __name__ == "__main__":
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results/Experiments_MLP', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    main()


