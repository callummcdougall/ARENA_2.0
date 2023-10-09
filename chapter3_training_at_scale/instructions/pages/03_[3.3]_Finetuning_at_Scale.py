
import os, sys
from pathlib import Path
chapter = r"chapter3_training_at_scale"
instructions_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/instructions").resolve()
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st
import st_dependencies

st_dependencies.styling()

import platform
is_local = (platform.processor() != "")

def section_0():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
    <li class='margtop'><a class='contents-el' href='#huggingface-accelerate'>Huggingface Accelerate</a></li>
    <li><ul class="contents">
        <li class='margtop'><a class='contents-el' href='#exercise-convert-into-distributed-training-loop-using-huggingface-accelerate'><b>Exercise</b> - Convert into distributed training loop using Huggingface Accelerate</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#microsoft-deepspeed'>Microsoft DeepSpeed</a></li>
    <li><ul class="contents">
        <li class='margtop'><a class='contents-el' href='#exercise-distributed-training-loop'><b>Exercise</b> - DeepSpeed training loop</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#trlx'>TRLX</a></li>
    <li><ul class="contents">
        <li class='margtop'><a class='contents-el' href='#exercise-trlx-distributed-training'><b>Exercise</b> - TRLX distributed training</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#bonus'>Bonus</a></li>
    <li><ul class="contents">
        <li class='margtop'><a class='contents-el' href='#finetuning-vanilla-gpt2-on-the-simulacra-dataset'>Finetuning vanilla GPT-2 on the simulacra dataset</a></li>
        <li class='margtop'><a class='contents-el' href='#train-anything-your-heart-desires'>Train anything your heart desires</a></li>
    </ul></li>

</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/finetuning.png" width="350">


Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [3.3] - Finetuning at Scale


## Introduction


In this section, we delve into the intricacies of finetuning neural networks at scale. We explore various strategies and methodologies that enable us to adapt pretrained models effectively and achieve state-of-the-art performance. 

We will start off by looking into some of the off-the-shelf distribute training libraries namely:

1. Huggingface Accelerate
2. Microsoft DeepSpeed
3. TRLX

We will be looking into common recipes for getting started with your own training loops and talking about the diffrentiating features of these libraries.

## Learning objectives

- Learning to use Huggingface Accelerate, DeepSpeed and TRLX
- Working with third-party training optimization libraries
- Develop the patience to wait for your training to finish (thanks Copilot for that one)

## Setup

```python
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

root_dir = exercises_dir.parent.parent.resolve()
if str(root_dir) not in sys.path: sys.path.append(str(root_dir))
os.chdir(root_dir)
from chapter2_rl.exercises.part4_rlhf.solutions import reward_model, ppo_config, prompts
os.chdir(orig_dir)
```

## Huggingface Accelerate


Huggingface Accelerate is a high-level library developed by Hugging Face, a leading provider of natural language processing (NLP) tools and models. Accelerate is designed to simplify and optimize the training and inference processes for deep learning models, particularly in the context of NLP tasks.

The primary goal of Huggingface Accelerate is to provide a user-friendly and efficient framework for distributed training. It aims to make it easier for researchers and practitioners to leverage multiple GPUs or even distributed computing setups to train their models faster and more effectively.

Accelerate achieves this by abstracting away the complexities of distributed training, allowing users to focus on model development and experimentation rather than low-level distributed computing details. It provides a simple and consistent interface that works across different deep learning frameworks, such as PyTorch and TensorFlow, and supports various distributed training strategies like data parallelism and model parallelism.

### Exercise - Convert into distributed training loop using Huggingface Accelerate

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 30-40 minutes on this exercise.
```

Take a look at the Huggingface documentation for [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/migration) and apply the recommended changes to turn a vanilla PyTorch loop into an Accelerate loop.

Below is the vanilla PyTroch training loop that you'll be modifying today:

```python
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
```
```python
def accelerate_train(model, train_dataset, num_epochs=10):
    pass
```

<details>
<summary>Solution </summary>

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
</details>

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

### Huggingface Accelerate Pros and Cons

Pros:

1. Simplified Distributed Training: Huggingface Accelerate provides a high-level abstraction for distributed training, making it easier to leverage multiple GPUs or distributed computing setups. It abstracts away the complexities of parallelization and synchronization, allowing researchers and practitioners to focus on model development and experimentation.

2. Framework Agnostic: Accelerate supports multiple deep learning frameworks, including PyTorch and TensorFlow. This flexibility allows users to work with their preferred framework and seamlessly switch between them, depending on their project requirements.

3. Efficient Data Loading: The library includes optimized data loading utilities, such as the DistributedDataLoader, which efficiently distribute and preprocess data across multiple processes or nodes. This feature ensures maximum data throughput during training, enhancing overall training efficiency.

4. Automatic Mixed Precision: Accelerate supports automatic mixed precision training, which takes advantage of GPU capabilities to perform calculations in lower-precision formats. This feature accelerates training without compromising numerical stability, leading to faster training times.

5. Experiment Tracking: Accelerate integrates well with the Hugging Face Trainer API, allowing easy tracking and logging of training metrics. This facilitates experiment management and comparison, making it simpler to analyze and reproduce results.

Cons:

1. Learning Curve: While Accelerate simplifies the process of distributed training, it still requires some understanding of distributed computing concepts. Users who are unfamiliar with distributed training may need to invest time in learning and understanding the library's concepts and usage.

2. Limited to Deep Learning: Huggingface Accelerate is primarily designed for deep learning tasks, particularly in the field of natural language processing. If you are working on non-deep learning tasks or outside the realm of NLP, other libraries or frameworks might be more suitable.

3. Dependency on Hugging Face Ecosystem: Accelerate is closely tied to the Hugging Face ecosystem, which means you may need to use other Hugging Face libraries or tools for certain functionalities or models. If you prefer a more modular approach or want to use different libraries or models, this dependency may limit your flexibility.

4. Performance Trade-offs: While Accelerate offers efficient distributed training, the performance gains might vary depending on the specific hardware and network setup. It's important to carefully evaluate the performance impact of distributed training and assess whether the gains justify the additional complexity.

5. Lack of Customization: While Accelerate provides a convenient and straightforward interface, it may lack certain customization options compared to lower-level frameworks. If you require fine-grained control over distributed training strategies or have unique requirements, you may find the abstraction of Accelerate limiting.

### Exercise - Huggingface Finetuning

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 30-40 minutes on this exercise.
```

All of the instructions for this exercise can be found [here](https://huggingface.co/docs/transformers/main/training#train-with-pytorch-trainer), this exercise will also test your aptitude with reading documentation and translating it into working code.

Task: Finetune BERT with the Yelp dataset to output Yelp reviews

Get the dataset from Huggingface hosted [here](https://huggingface.co/datasets/yelp_review_full), we will be using the BERT model hosted [here](https://huggingface.co/bert-base-cased).

We will also briefly talk about the Huggingface Trainer object:

The (Trainer)[https://huggingface.co/docs/transformers/main_classes/trainer#trainer] class has three arguments that are essential to starting any training run which are:

1. model - The model that you want to train which could either be a PyTorch model or a pretrained Transformers model. For this exercise we will be using a Transformers model hosted [here](https://huggingface.co/bert-base-uncased)
2. args - The args is an object of the [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) class that will contain all the hyperparameters the Trainer will use for training and/or evaluation.
3. train_dataset - The train_dataset is a Huggingface dataset object

Additionally you might want to add arguments if you want to work with other models especially language transformers:

1. eval_dataset - The dataset to use for evaluation
2. tokenizer - The tokenizer used to preprocess the data

Things to note:

1. We want to move to a model from Huggingface Transformers and ditch our old torchvision model, this is due to the fact that the Huggingface Trainer plays nicely with the models in the Transformers library.
2. We need to use a compute_metrics function that will do the training evaluations for us during the training run

<details>
<summary>Solution</summary>

```python
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
```

</details>

## Microsoft DeepSpeed

Microsoft DeepSpeed is a library and optimization engine designed to accelerate and scale deep learning training on large-scale models. It focuses on reducing memory consumption and increasing throughput. DeepSpeed implements everything in the [ZeRo](https://arxiv.org/pdf/1910.02054.pdf) paper and is worth a read to understand the specific optimisations implemented in the library. 
Out of the optimisations bundled in with DeepSpeed the following are the most notable:

1. ZeRO Memory Optimization: DeepSpeed introduces the ZeRO (Zero Redundancy Optimizer) technique, which optimizes memory consumption during training. ZeRO allows training models that are much larger than the GPU memory capacity by partitioning and optimizing memory usage across multiple devices.

2. Activation Checkpointing: DeepSpeed implements activation checkpointing, a technique that trades compute time for memory consumption. It selectively recomputes activations on the fly during backward passes, reducing the memory footprint of large models and enabling the training of larger models on limited memory resources.

3. Offload Optimizations: DeepSpeed leverages mixed-precision training and tensor offloading to reduce compute time and memory utilization. It offloads computation to the CPU or lower-precision hardware, such as tensor cores, to speed up training and conserve GPU memory.

4. Pipeline Parallelism: The library supports pipeline parallelism, a technique for distributing large models across multiple GPUs or devices. It partitions models into stages and parallelizes the computation, enabling training of extremely large models that would otherwise exceed the memory capacity of individual GPUs.

5. Gradient Compression: DeepSpeed incorporates gradient compression algorithms to reduce the communication overhead during distributed training. It uses techniques like gradient accumulation and quantization to compress gradients, enabling efficient gradient exchange and improving scalability for distributed training.

6. Automatic Loss Scaling: DeepSpeed provides automatic loss scaling, a technique that mitigates numerical instability issues associated with training in lower-precision formats. It dynamically adjusts the scaling factor for gradients, ensuring stable training with mixed-precision calculations.

7. Integration with PyTorch Ecosystem: DeepSpeed is designed to seamlessly integrate with the PyTorch ecosystem. It can be easily integrated into existing PyTorch codebases and is compatible with various PyTorch libraries, models, and optimization techniques.

Huggingface Accelerate comes prepackaged with Microsoft DeepSpeed optimisations and the only way to use them in a Jupyter notebook is through the (Huggingface Trainer class)[https://huggingface.co/docs/transformers/main_classes/trainer]. 

### Exercise - Huggingface Trainer class
```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 20-30 minutes on this exercise.
```

```python

def huggingface_train_with_Trainer():
 	## Initialise model and training dataset here
 	model = ...
  	train_dataset = ...
 
 	training_args = TrainingArguments(...) # fill in hyperparameters similar to previous training runs
	
 	trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
	trainer.train()
```

<details>
<summary>Solution</summary>


```python

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
    evaluation_strategy="epoch"
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
```
</details>

### Exercise - DeepSpeed training loop

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 30-40 minutes on this exercise.
```

Create a DeepSpeed training loop which mimics the properties of the earlier training loops.

This will involve creating a new file, the config file which we will name config ds_config.json. To create a config file from the jupyter notebook start with:

```python

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
```

Examples of how config files can look like can be found [here](https://huggingface.co/docs/transformers/main_classes/deepspeed#deployment-in-notebooks) and the optimisations available on DeepSpeed can be found [here](https://www.deepspeed.ai/training/#features). 
Try crafting a config file that works with as many optimisations as possible. The config file in the solutions is just a guide and not the best possible config file.

We also need to make some changes in our previous training loop to incorporate the deepspeed config, your training function should look something like this:

```python

# DeepSpeed requires a distributed environment even when only one process is used.
# This emulates a launcher in the notebook
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
```
Copy in the arguments from your prior loop into this loop for all other empty spaces.

<details>
<summary>Solution</summary>

Config file:
```python
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

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

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}

```
</details>

## TRLX

We've encountered TRLX before and it uses Huggingface Accelerate as the backend. This means we should be able to directly apply what we did in the first section today to get TRLX started with distributed training.

The magic that turns our single GPU TRLX loop into an Acceerate distributed loop is here:

```python
accelerator = Accelerator()
accelerator.wait_for_everyone()
```

### Exercise - TRLX distributed training

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 40-50 minutes on this exercise.
```

Copy in your training loops from the RLHF sections of the RL chapter and add the magic code in to turn your code into distributed training code which should work simply out of the box. There need to be extra setup steps before you use TRLX because it expects you to have different versions of the transformers library than what we've been using today.

Setup:

```python
In terminal:

git clone https://github.com/atagade/trlx
cd trlx
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu116 # for cuda
pip install -e .

In notebook:

from trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from trlx.models.modeling_ppo import PPOConfig
from trlx import train
```

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

### Finetuning vanilla GPT-2 on the [simulacra dataset](https://github.com/JD-P/simulacra-aesthetic-captions)

Since this dataset is quite large, the usage of distributed training will enable us to get the finetuning done quicker. You can use this python example [file](https://github.com/CarperAI/trlx/blob/main/examples/simulacra.py) as inspiration.

### Train anything your heart desires

With the technqiues you've learnt in this section we can train any model on any dataset (preferably it's hosted on Huggingface). If you're here and thinking about more things to do, find a research paper you've always admired and try to replicate their results on a distributed setup to show if the assertions of the paper hold at scale.



















""", unsafe_allow_html=True)

section_0()
