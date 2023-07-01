#%%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from accelerate import Accelerator
from pathlib import Path
import time
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification

orig_dir = os.getcwd()

chapter = r"chapter3_training_at_scale"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

# root_dir = exercises_dir.parent.parent.resolve()
# if str(root_dir) not in sys.path: sys.path.append(str(root_dir))
# os.chdir(root_dir)
# from chapter2_rl.exercises.part4_rlhf.solutions import reward_model, ppo_config, prompts
# os.chdir(orig_dir)

#%%
def train(model, train_dataset, num_epochs=10):

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Define your training data and labels
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over the training dataset
        for inputs, labels in train_loader:
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training finished.")

def accelerate_train(model, train_dataset, num_epochs=10):
    accelerator = Accelerator()
    device = accelerator.device

    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Define your training data and labels
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # model.to(device)
    
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader#, scheduler
    )

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            # scheduler.step()

            # Update running loss
            running_loss += loss.item()


        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training finished.")


# %%

model = torchvision.models.resnet18()
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
train_dataset = torchvision.datasets.CIFAR100(root='/data/', download=True, train=True, transform=transform_train)

# start_time = time.time()
# train(model, train_dataset, num_epochs=5)
# end_time = time.time()

# print(f'Time taken for vanilla training = {end_time -start_time} seconds')

# start_time = time.time()
# accelerate_train(model, train_dataset, num_epochs=5)
# end_time = time.time()

# print(f'Time taken for Accelerate training = {end_time -start_time} seconds')
# %%

from transformers import AutoTokenizer, AutoModelForSequenceClassification, 


def huggingface_train_with_Trainer():
 	## Initialise model and training dataset here
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

    raw_dataset = load_dataset("glue", "mrpc")

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    train_dataset = torchvision.datasets.CIFAR100(root='/data/', download=True, train=True, transform=transform_train)
    
    training_args = TrainingArguments(
        output_dir='output',
        num_train_epochs=10,
        optim="sgd",
        learning_rate=0.001
    ) # fill in hyperparameters similar to previous training runs

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()


huggingface_train_with_Trainer()
# %%