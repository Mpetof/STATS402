import torch
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from model_construction import GCNPair, map_labels, load_cpkl_gz, load_data
import torch.nn.functional as F

# Request user input for configurable parameters
epochs = int(input("Enter the number of epochs: "))
learning_rate = float(input("Enter the learning rate: "))
hidden_dim = int(input("Enter the hidden dimension size: "))
model_save_name = input("Enter the model save name (e.g., GCN_node_model.pth): ")

# Device Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

def train_one_epoch(model, optimizer, criterion, train_data, epoch, device):
    total_loss = 0
    correct = 0
    total = 0
    model.train()
    for pair_id in range(len(train_data[1])):
        graph_l, graph_r, label = load_data('train', pair_id, train_data, device)
        label_mapped = map_labels(label[:, -1])
        label_tensor = torch.tensor(label_mapped, dtype=torch.long).to(device)
        optimizer.zero_grad()
        out = model(graph_l, graph_r, label)
        loss = criterion(out, label_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == label_tensor).sum().item()
        total += label_tensor.size(0)
    avg_loss = total_loss / len(train_data[1])
    train_acc = correct / total
    print(f'Epoch: {epoch:03d}, Loss: {avg_loss:.5f}, Accuracy: {train_acc:.5f}')
    return avg_loss, train_acc

def test(model, criterion, test_data, device):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    correct = 0
    total = 0
    with torch.no_grad():
        for pair_id in range(len(test_data[1])):
            graph_l, graph_r, label = load_data('test', pair_id, test_data, device)
            label_mapped = map_labels(label[:, -1])
            label_tensor = torch.tensor(label_mapped, dtype=torch.long).to(device)

            out = model(graph_l, graph_r, label)
            pred = out.argmax(dim=1)
            prob = F.softmax(out, dim=1)[:, 1]

            y_true.extend(label_tensor.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_scores.extend(prob.cpu().numpy())

            correct += (pred == label_tensor).sum().item()
            total += label_tensor.size(0)
    test_acc = correct / total
    return test_acc, y_true, y_pred, y_scores

def plot_training_graphs(epochs, train_losses, train_accuracies, model_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy Over Epochs')
    plt.legend()

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

def plot_roc_curve(y_true, y_scores, model_name):
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
    plt.legend()
    plt.savefig(f"./results/{model_name}_roc_curve.pdf")
    plt.show()


def main():
    train_data = load_cpkl_gz('./dataset/train.cpkl.gz')
    test_data = load_cpkl_gz('./dataset/test.cpkl.gz')

    model = GCNPair(input_dim=70, hidden_dim=hidden_dim, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weight = torch.tensor([1, 5], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    train_losses, train_accuracies = [], []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, optimizer, criterion, train_data, epoch, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
    end_training_time = time.time()

    test_accuracy, y_true, y_pred, y_scores = test(model, criterion, test_data, device)
    end_testing_time = time.time()

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f"Training took {end_training_time - start_time:.2f} seconds.")
    print(f"Inference took {end_testing_time - end_training_time:.2f} seconds.")

    torch.save(model.state_dict(), f'./models/{model_save_name}')

    # Outputs for plotting
    #print(f"Training Losses: {train_losses}")
    #print(f"Training Accuracies: {train_accuracies}")
    #print(f"Test True Labels: {y_true}")
    #print(f"Test Predicted Labels: {y_pred}")
    #print(f"Test Scores (Probabilities): {y_scores}")

    model_base_name = model_save_name.replace('.pth', '') 
    plot_training_graphs(range(1, epochs + 1), train_losses, train_accuracies, model_base_name)
    plot_confusion_matrix(y_true, y_pred, model_base_name)
    plot_roc_curve(y_true, y_scores, model_base_name)

if __name__ == "__main__":
    main()
