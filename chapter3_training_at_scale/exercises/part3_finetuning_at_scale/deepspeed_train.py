import deepspeed
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

model = torchvision.models.resnet18()

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()

model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=model.parameters())

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

train_dataset = torchvision.datasets.CIFAR100(root='/data/', download=True, train=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

num_epochs = 5

for epoch in num_epochs:
  
  epoch_loss = 0

  for step, batch in enumerate(train_loader):
    #forward() method
    loss = model_engine(batch)
    epoch_loss += loss

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
    
  print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print('DeepSpeed training finished')