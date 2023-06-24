# %%
import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
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
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
# root_dir = exercises_dir.parent.parent.resolve()
# if str(root_dir) not in sys.path: sys.path.append(str(root_dir))
# os.chdir(root_dir)
# from chapter2_rl.exercises.part4_rlhf.solutions import reward_model, ppo_config, prompts
# os.chdir(orig_dir)
# %%
def train(model, train_dataset, num_epochs=10):
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # Define your training data and labels
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
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
    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # Define your training data and labels
    training_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    model, optimizer, training_dataloader = accelerator.prepare(
        model, optimizer, training_dataloader
    )
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Iterate over the training dataset
        for inputs, labels in training_dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass
            accelerator.backward(loss)
            # Update weights
            optimizer.step()
            # Update running loss
            running_loss += loss.item()
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(training_dataloader)
        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    print("Training finished.")
# %%
# model = torchvision.models.resnet18()
# transform_train = transforms.Compose(
#     [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ]
# )
# train_dataset = torchvision.datasets.CIFAR100(
#     root="/data/", download=True, train=True, transform=transform_train
# )
# start_time = time.time()
# train(model, train_dataset, num_epochs=5)
# end_time = time.time()
# print(f"Time taken for vanilla training = {end_time -start_time} seconds")
# start_time = time.time()
# accelerate_train(model, train_dataset, num_epochs=5)
# end_time = time.time()
# print(f"Time taken for Accelerate training = {end_time -start_time} seconds")
# %%
# %%
import numpy as np
import evaluate
from datasets import load_dataset
def compute_metrics(eval_pred, metric_type="accuracy"):
    metric = evaluate.load(metric_type)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
# %%
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import gc
gc.collect()
torch.cuda.empty_cache()
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
# %%
def huggingface_train_with_Trainer():
    ## Initialise model and training dataset here
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5
    )
    training_args = TrainingArguments(
        output_dir="yelp",
        do_train=True,
        num_train_epochs=1,
        evaluation_strategy="epoch",  # fill in hyperparameters similar to previous training runs
    )
    training_args.set_training(learning_rate=1e-5, batch_size=8, weight_decay=0.01)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
# %%
# huggingface_train_with_Trainer()
# %%
# %%
# DeepSpeed requires a distributed environment even when only one process is used.
# This emulates a launcher in the notebook
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5
    )
training_args = TrainingArguments(
    output_dir="yelp_ds",
    do_train=True,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    deepspeed="ds_config.json"
    # fill in hyperparameters similar to previous training runs
)
training_args.set_training(learning_rate=1e-5, batch_size=8, weight_decay=0.01)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
# Now proceed as normal, plus pass the deepspeed config file
trainer.train()
# %%
