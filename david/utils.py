# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
import torch.nn.utils.prune as prune
import numpy as np
import os
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
# %%
class LeNetUltraSparse(nn.Module):
    def __init__(self):
        super(LeNetUltraSparse, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.fc1 = nn.Linear(3 * 6 * 6, 10)  # 5x5 image dimension
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x

# model = LeNetUltraSparse()
# model.to(device)
# summary(model, input_size=(7,1,28,28))

# %%

class LeNetSparse(nn.Module):
    def __init__(self):
        super(LeNetSparse, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 10)  # 5x5 image dimension
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x

# %%
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
class LeNetv2(nn.Module):
    def __init__(self):
        super(LeNetv2, self).__init__()

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

class LeNetv3(nn.Module):
    def __init__(self):
        super(LeNetv3, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 10)
       
        self.activations = {}

    def forward(self, x):
    
        
        x = self.conv1(x)
        self.activations['conv1'] = x.clone()
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        self.activations['pre_conv2'] = x.clone()

        x = self.conv2(x)
        self.activations['conv2'] = x.clone()
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        self.activations['pre_fc1'] = x.clone()

        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        self.activations['fc1'] = x.clone()
        x = F.relu(x)
        self.activations['pre_fc2'] = x.clone()

        x = self.fc2(x)
        
        return x
# %%

class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"Elapsed time: {self.elapsed_time:.4f} sec")

# %%
force_download = True

# Check if the saved files exist
all_files = all(os.path.exists(file) for file in ["train_data.pt", "train_labels.pt", "test_data.pt", "test_labels.pt"]) 

if all_files and not force_download:
    # Load the data from disk
    train_data = torch.load("train_data.pt").to(device)
    train_labels = torch.load("train_labels.pt").to(device)
    test_data = torch.load("test_data.pt").to(device)
    test_labels = torch.load("test_labels.pt").to(device)
else:
    train_dataset = MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
    ]))
    test_dataset = MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
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