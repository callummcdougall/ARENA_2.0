# %%

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

# %%

def train(model, train_dataset, num_epochs=10):

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# %%

def accelerate_train(model, train_dataset, num_epochs=10):
    # SOLUTION
    accelerator = Accelerator()

    device = accelerator.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Define your training data and labels
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Accelerator prepare
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over the training dataset
        for inputs, labels in train_loader:

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            accelerator.backward(loss) #loss.backward()

            # Update weights
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training with Huggingface Accelerate finished.")

# %%

model = torchvision.models.resnet18()
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
train_dataset = torchvision.datasets.CIFAR100(root='/data/', download=True, train=True, transform=transform_train)

start_time = time.time()
train(model, train_dataset, num_epochs=5)
end_time = time.time()

print(f'Time taken for vanilla training = {end_time -start_time} seconds')

start_time = time.time()
accelerate_train(model, train_dataset, num_epochs=5)
end_time = time.time()

print(f'Time taken for Accelerate training = {end_time -start_time} seconds')

# %%

raw_dataset = load_dataset("glue", "mrpc")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
  return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments('test')

trainer = Trainer(
                  model,
                  training_args,
                  train_dataset = tokenized_datasets['train'],
                  eval_dataset = tokenized_dataset['validation'],
                  data_collator = data_collator,
                  tokenizer = tokenizer
)

trainer.train()

# %%

%%bash
cat <<'EOT' > ds_config.json

{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
}

# %%

import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# Now proceed as normal, plus pass the deepspeed config file
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()