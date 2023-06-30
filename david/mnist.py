# %%
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import utils
import torch.nn.utils as tutils
from torchinfo import summary
# %%

# cpu_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, pin_memory_device='cuda')
# cpu_test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, pin_memory=True, pin_memory_device='cuda')

run_test_loss = []
run_train_loss = []
run_test_acc  = []
run_reg_weight = []
# %%

def train(model, num_epochs = 20, reg_weight = 0.0002, lr=0.001):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    runner = tqdm(range(num_epochs))
    for epoch in runner:
        # Training
        train_loss = 0.0
        avg_reg_loss = 0.0
        model.train()
        for images, labels in utils.train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_data = criterion(outputs, labels)
            #add L2 regularation to the model   

            loss_reg = torch.norm(tutils.parameters_to_vector(model.parameters()),p=1)
            
            loss = loss_data + reg_weight * loss_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            avg_reg_loss += loss_reg.item() * reg_weight * images.size(0)

        # Testing
        test_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in utils.test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(utils.train_loader.dataset)
        avg_reg_loss /= len(utils.train_loader.dataset) 
        test_loss /= len(utils.test_loader.dataset)
        test_accuracy = 100.0 * correct / total
        run_test_loss.append(test_loss)
        run_train_loss.append(train_loss)
        run_test_acc.append(test_accuracy)
        run_reg_weight.append(avg_reg_loss)
        runner.set_description(f"Epoch {epoch + 1}/{num_epochs} - Train {train_loss:.4f} - Reg {avg_reg_loss:.4f} - Test: {test_loss:.4f} - Acc: {test_accuracy:.4f}%")
        if test_accuracy > 98.9:
            break
# %%
model = utils.LeNetv2()
model.to(device)
summary(model, input_size=(7,1,28,28))
# %%
train(model, num_epochs=20, reg_weight=0.0001, lr=0.0003)

# %%
plt.plot(run_test_loss, label='Test Loss')
plt.plot(run_train_loss, label='Train Loss')
plt.plot(run_reg_weight, label='Reg Loss')
plt.plot()
plt.legend()
plt.yscale('log')
plt.show()
# %%
torch.save(model.state_dict(), 'LeNet_Fashion_9045.pth')

# %%
