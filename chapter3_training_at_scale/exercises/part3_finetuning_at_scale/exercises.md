# Chapter 3 - Finetuning At Scale

In this chapter, we delve into the intricacies of finetuning neural networks at scale. We explore various strategies and methodologies that enable us to adapt pretrained models effectively and achieve state-of-the-art performance. 
We will start off by looking into some of the off-the-shelf distribute training libraries namely:

1. Huggingface Accelerate
2. Microsoft Deepspeed
3. TRLX

We will be looking into common recipes for getting started with your own training loops and talking about the diffrentiating features of these libraries.

Setup:
```python

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from accelerate import Accelerator

from trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from trlx.models.modeling_ppo import PPOConfig
from trlx import train

orig_dir = os.getcwd()

chapter = r"chapter3_training_at_scale"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

root_dir = exercises_dir.parent.parent.resolve()
if str(root_dir) not in sys.path: sys.path.append(str(root_dir))
os.chdir(root_dir)
from chapter2_rl.exercises.part4_rlhf.solutions import reward_model, ppo_config, prompts
os.chdir(orig_dir)

```

## Huggingface Accelerate

Below is a standard training loop that you've most likely seen before:

```python
def train():

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define your model
    model = torchvision.models.resnet18().to(device)

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
```

### Exercise - Convert the above training loop into a distributed training loop using Huggingface Accelerate

Difficulty: 3/5
Importance: 4/5

This should take you 30-40 minutes to complete.

Take a look at the Huggingface documentation for [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/migration) and apply the recommended changes to turn a vanilla PyTorch loop into an Accelerate loop.

# SOLUTION
```python
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

```

We'll use the following code to test the runtimes of the two models:

```python
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
```

We don't use Huggingface Accelerate for speed, we use it for convenience. The difference in the number of lines in our training functions above is quite small but when you move to a distributed setup, the resent_train.py function which contains vanilla distributed tranining code will be many times larger than the Huggingface training loop above.

## Microsoft DeepSpeed

DeepSpeed was built with training efficiency in mind and provides a production-ready library experience. The full list of features that come with DeepSpeed can be seen [here](https://www.deepspeed.ai/training/). We also need to move to a production ML folder structure that looks something like this:

```
- project_folder/
  - dataset/
    - train/
    - validation/
    - test/
  - models/
    - deepspeed/
      - model.py
      - model_config.json
  - scripts/
    - train.py
    - evaluate.py
    - utils.py
  - checkpoints/
  - logs/
  - requirements.txt
```

For the purposes of the following exercise we won't be moving to the above folder but will be creating a config file that contains the parameters of the trianing run that we wish to execute. A sample config file looks likes this:

```
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
}
```

### Exercise: Create a deepspeed training loop that mimics the properties of the earlier training loops

Difficulty: 4/5
Importance: 4/5

This should take you 30-40 minutes to complete.

Create two new files, the config file and the training script. The training script should look something like this:

```python
import deepspeed
import argparse


parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()

model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=net,
                                                     model_parameters=net.parameters())

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
```

Make sure your config file looks similar to the previous training runs and add a timing element to the script above to benchmark the training time with the other frameworks. You can try different training optimisations by going [here](https://www.deepspeed.ai/training/) and modifying the config file appropriately.

```
# SOLUTION

config file:

{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "optimizer": {
      "type": "SGD",
      "params": {
        "lr": 0.001
      }
    },
    "zero_optimization": False
  }

```

## Exercise: Create a deepspeed training loop with training optimisations

The config is the best place to add in these training optimisations, refer to the list of available speedups [here](https://www.deepspeed.ai/training/).

An example config for this could look like:

```python
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": true
}
```

Try out different optimisations to see the decrease in training times! It should be as simple as adding arguments in the config file. There might be optimisations that are incompatible, try to think carefully about which optimisations complement each other and which don't.


## TRLX

We've encountered TRLX before and it uses Huggingface Accelerate as the backend. This means we should be able to directly apply what we did in the first section today to get TRLX started with distributed training.

The magic that turns our single GPU TRLX loop into an Acceerate distributed loop is here:

```python
accelerator = Accelerator()
accelerator.wait_for_everyone()
```

### Exercise - TRLX distributed training

Difficulty: 3/5
Importance: 4/5

This should take you 40-50 minutes to complete.

Copy in your training loops from the RLHF sections of the RL chapter and add the magic code in to turn your code into distributed training code which should work simply out of the box.

```python
# SOLUTION
accelerator = Accelerator()
accelerator.wait_for_everyone()

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
```

## Bonus

### Finetuning a vanilla GPT-2 model on the [simulacra dataset](https://github.com/JD-P/simulacra-aesthetic-captions)

Since this dataset is quite large, the usage of distributed training will enable us to get the finetuning done quicker. You can use this python example [file](https://github.com/CarperAI/trlx/blob/main/examples/simulacra.py) as inspiration.

### Train anything your heart desires

With the technqiues you've learnt in this section we can train any model on any dataset (preferably it's hosted on Huggingface). If you're here and thinking about more things to do, find a research paper you've always admired and try to replicate their results on a distributed setup to show if the assertions of the paper hold at scale.