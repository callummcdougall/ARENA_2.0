# %%
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class GPUDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.to('cuda')  # Move the data to the GPU
        self.labels = labels.to('cuda')  # Move the labels to the GPU

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        label_item = self.labels[index]
        return data_item, label_item

def compute_acc(model):
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    _, gpu_mnist_test = load_data()

    images, labels = gpu_mnist_test.data, gpu_mnist_test.labels

    model.eval()
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    test_loss /= len(gpu_mnist_test)
    test_accuracy = 100.0 * correct / total
    return test_accuracy, test_loss

def init_data():

    train_dataset = MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_dataset = MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]))

    cpu_train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    cpu_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    mnist_train_data = []
    mnist_train_labels = []

    for x, y in cpu_train_loader:
        mnist_train_data.append(x)
        mnist_train_labels.append(y)

    mnist_train_data = torch.cat(mnist_train_data, dim=0)
    mnist_train_labels = torch.cat(mnist_train_labels, dim=0)

    mnist_test_data = []
    mnist_test_labels = []

    for x, y in cpu_test_loader:
        mnist_test_data.append(x)
        mnist_test_labels.append(y)

    mnist_test_data = torch.cat(mnist_test_data, dim=0)
    mnist_test_labels = torch.cat(mnist_test_labels, dim=0)

    torch.save(mnist_train_data, 'mnist_train_data.pt')
    torch.save(mnist_train_labels, 'mnist_train_labels.pt')
    torch.save(mnist_test_data, 'mnist_test_data.pt')
    torch.save(mnist_test_labels, 'mnist_test_labels.pt')

def load_data():
    mnist_train_data = torch.load('mnist_train_data.pt').to(device)
    mnist_train_labels = torch.load('mnist_train_labels.pt').to(device)
    mnist_test_data = torch.load('mnist_test_data.pt').to(device)
    mnist_test_labels = torch.load('mnist_test_labels.pt').to(device)

    gpu_mnist_train = GPUDataset(mnist_train_data, mnist_train_labels)
    gpu_mnist_test = GPUDataset(mnist_test_data, mnist_test_labels)

    return gpu_mnist_train, gpu_mnist_test
# %%

import numpy as np
import matplotlib.pyplot as plt

def plot_tensor_grid(tensor, titles=None):
    num_images = tensor.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(tensor[i].cpu().squeeze())
            ax.axis('off')
            
            if titles is not None:
                ax.set_title(titles[i].item())
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()

# %%
