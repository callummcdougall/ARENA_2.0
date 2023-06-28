# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
import torch.nn.utils.prune as prune
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 10)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x
# %%

# Check if the saved files exist
if os.path.exists("train_data.pt") and os.path.exists("train_labels.pt") and os.path.exists("test_data.pt") and os.path.exists("test_labels.pt"):
    # Load the data from disk
    train_data = torch.load("train_data.pt").to(device)
    train_labels = torch.load("train_labels.pt").to(device)
    test_data = torch.load("test_data.pt").to(device)
    test_labels = torch.load("test_labels.pt").to(device)
else:
    train_dataset = MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_dataset = MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]))

    # Compute the data and save it to disk
    train_data = torch.stack([x[0] for x in train_dataset]).to(device)
    train_labels = torch.tensor([x[1] for x in train_dataset], device=device)
    test_data = torch.stack([x[0] for x in test_dataset]).to(device)
    test_labels = torch.tensor([x[1] for x in test_dataset], device=device)

    # Save the data to disk
    torch.save(train_data, "train_data.pt")
    torch.save(train_labels, "train_labels.pt")
    torch.save(test_data, "test_data.pt")
    torch.save(test_labels, "test_labels.pt")

# %%
gpu_train_dataset = TensorDataset(train_data, train_labels)
gpu_test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(gpu_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(gpu_test_dataset, batch_size=10000, shuffle=False)

def compute_acc(model):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / total
    return test_accuracy, test_loss



# %%