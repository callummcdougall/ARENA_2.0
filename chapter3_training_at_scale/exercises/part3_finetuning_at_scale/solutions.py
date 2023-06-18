# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
import time

import deepspeed
import argparse

from trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from trlx.models.modeling_ppo import PPOConfig
from trlx import train

import os
root_dir = exercises_dir.parent.parent.resolve()
if str(root_dir) not in sys.path: sys.path.append(str(root_dir))
os.chdir(root_dir)
from chapter0_fundamentals.exercises.part3_resnets.solutions import ResNet34
os.chdir(orig_dir)

# %%
def train(model, train_dataset, num_epochs=10):

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Set model to device
    model = model.to(device)

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

## Exercise - Convert the above training loop into an Accelerate training loop

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
accelerator = Accelerator()
accelerator.wait_for_everyone()

def ppo_config():
	return TRLConfig(
		train=TrainConfig(
			seq_length=1024,
			epochs=100,
			total_steps=10000,
			batch_size=32,
			checkpoint_interval=10000,
			eval_interval=100,
			pipeline="PromptPipeline",
			trainer="AcceleratePPOTrainer",
		),
		model=ModelConfig(model_path="lvwerra/gpt2", num_layers_unfrozen=2),
		tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
		optimizer=OptimizerConfig(
			name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
		),
		scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
		method=PPOConfig(
			name="PPOConfig",
			num_rollouts=128,
			chunk_size=128,
			ppo_epochs=4,
			init_kl_coef=0.001,
			target=None,
			horizon=10000,
			gamma=1,
			lam=0.95,
			cliprange=0.2,
			cliprange_value=0.2,
			vf_coef=1,
			scale_reward="ignored",
			ref_mean=None,
			ref_std=None,
			cliprange_reward=10,
			gen_kwargs=dict(
				max_new_tokens=64,
				top_k=10,
				#top_p=1.0,
				do_sample=True,
			),
		),
	)

def main() -> None:
	# solution
	config = ppo_config()

	trlx.train(
		reward_fn = reward_model,
		prompts = prompts,
		eval_prompts = ['In my opinion'] * 256, ## Feel free to try different prompts
		config =  config
	)


if MAIN:
	gc.collect()
	t.cuda.empty_cache()
	main()

# %%
