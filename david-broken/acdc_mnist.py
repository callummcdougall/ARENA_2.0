# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchinfo import summary
from utils import LeNet, load_data
import torchvision.transforms as transforms

# %%
#testset, trainset = load_data()

train_dataset = MNIST('./data', train=True, download=True, transform=transforms.Compose([
transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))
test_dataset = MNIST('./data', train=False, transform=transforms.Compose([
transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
]))


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

# Initialize the model
model = LeNet()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()



# %%
# Training loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


run_test_loss = []
run_train_loss = []
run_test_acc  = []

# %%
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 3
runner = tqdm(range(num_epochs))

for epoch in runner:
    # Training
    train_loss = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)

    # Testing
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Testing"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    train_loss /= len(trainset)
    test_loss /= len(testset)
    test_accuracy = 100.0 * correct / total
    run_test_loss.append(test_loss)
    run_train_loss.append(train_loss)
    run_test_acc.append(test_accuracy)
    runner.set_description(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {train_loss:.4f} - Testing loss: {test_loss:.4f} - Testing accuracy: {test_accuracy:.2f}%")
    
# %%

plt.plot(run_test_loss, label='Test loss')
#plt.plot(run_train_loss, label='Train loss')

plt.yscale('log')
plt.legend()
plt.show()

plt.plot(run_test_acc, label='Test accuracy')
plt.show()
# Save the trained model
torch.save(model.state_dict(), 'mnist_cnn_max_pool.pth')
# %%