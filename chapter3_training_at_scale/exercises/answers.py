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



from accelerate import Accelerator

def accelerate_train(model, train_dataset, num_epochs=10):

    # Set device (GPU or CPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator()
    device = accelerator.device

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Define your training data and labels
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    ) # Accelerate will only prepare objects that inherit from their respective PyTorch classes (such as torch.optim.Optimizer).
    # scheduler if applicable

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over the training dataset
        for inputs, labels in train_loader:
            # Move inputs and labels to the device
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            # Accelerate already moves things to the correct device

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            # loss.backward()
            accelerator.backward(loss)

            # Update weights
            optimizer.step()

            # scheduler.step() if applicable

            # Update running loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training with Huggingface Accelerate finished.")


# model = torchvision.models.resnet18()
# transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
# train_dataset = torchvision.datasets.CIFAR100(root='/data/', download=True, train=True, transform=transform_train)

# start_time = time.time()
# train(model, train_dataset, num_epochs=5)
# end_time = time.time()

# print(f'Time taken for vanilla training = {end_time -start_time} seconds')

# start_time = time.time()
# accelerate_train(model, train_dataset, num_epochs=5)
# end_time = time.time()

# print(f'Time taken for Accelerate training = {end_time -start_time} seconds')





# %%

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

# tokenizer & model for bert
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# yelp dataset
dataset = load_dataset("yelp_review_full")
dataset["train"][100]

# modify dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# dataset train-test-split
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# hyperparams
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs = 14.3538)

# evaluation
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
# %%
# trainer.train()
# model.save_pretrained('yelp_bert')

#%%%
model = AutoModelForSequenceClassification.from_pretrained("yelp_bert", num_labels = 5)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
text = "I love this restaurant!!!!!!!!!! :)"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
#%%

import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

def huggingface_train_with_Trainer():

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    
    # Initialise model and training dataset here
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    # Pass the deepspeed config file
    training_args = TrainingArguments(output_dir="huggingface_trainer", 
                                    evaluation_strategy="epoch",
                                    num_train_epochs=1,
                                    learning_rate=5e-4,
                                    auto_find_batch_size=True,
                                    weight_decay=0.01,
                                    deepspeed="ds_config.json"
                                    ) # fill in hyperparameters similar to previous training runs

    trainer = Trainer(model=model, args=training_args, train_dataset=small_train_dataset,
                    eval_dataset=small_eval_dataset, compute_metrics=compute_metrics)
    trainer.train()

huggingface_train_with_Trainer()    

# before, you weren't doing hella multi-GPU
# but now you are











#%%

from trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from trlx.models.modeling_ppo import PPOConfig
from trlx import train

# SOLUTION
accelerator = Accelerator()
accelerator.wait_for_everyone()

def reward_model(samples, **kwargs):
    '''
    Returns the rewards for the given samples, using the reward model `model`.
    
    kwargs are passed to your model during a forward pass.
    '''
    # SOLUTION
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    model =  AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")

    rewards = []
    
    inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt")
    
    with t.inference_mode():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], **kwargs)
    
    logits = outputs.logits
    probabilities = t.softmax(logits, dim=1)

    for reward in probabilities:
       rewards.append(reward[1].item())
    
    return rewards


# provided
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
		model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=2),
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

imdb = load_dataset("imdb", split="train+test")

def generate_prompts(dataset):
	prompts = [" ".join(review.split()[:4]) for review in dataset["text"]]
	return prompts

prompts = generate_prompts(imdb)

def main() -> None:
	# solution
	config = ppo_config()

	train(
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
