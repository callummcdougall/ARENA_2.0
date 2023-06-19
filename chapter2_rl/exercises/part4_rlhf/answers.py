# %%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import json
import sys
import math
import gc
from pathlib import Path
import torch as t
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,  AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertForMaskedLM
import logging
from typing import cast, Any, List, Optional, Union, Tuple

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_rlhf"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part4_rlhf.tests as tests
import part4_rlhf.utils as utils
from trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from trlx.models.modeling_ppo import PPOConfig
from trlx import train
# %%
bert = utils.load_pretrained_bert()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def predict(model: BertForMaskedLM, tokenizer: AutoTokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''

    # Make sure we're in eval mode
    model.eval()

    # Tokenizer returns a bunch of special BERT-specific things, we just want input ids
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    # Get top predictions at all the places we masked
    out = model(input_ids).logits
    preds = out[input_ids == tokenizer.mask_token_id]
    tops = preds.topk(k, dim=-1).indices

    return [[tokenizer.decode(t) for t in mask] for mask in tops]


your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
predictions = predict(bert, tokenizer, your_text)
print("Model predicted: \n", "\n".join(map(str, predictions)))
# %%
imdb = load_dataset("imdb", split="train+test")
# %%
def label_split(dataset) -> Tuple[int, int]:
    n_pos, n_neg = 0, 0
    
    for sample in dataset:
        if sample['label'] == 1:
            n_pos += 1
        if sample['label'] == 0: 
            n_neg += 1
            
    return n_pos, n_neg

n_pos, n_neg = label_split(imdb)

tests.test_label_split(n_pos, n_neg)
# %%
def generate_prompts(dataset) -> List[str]:
    '''Generate & return prompts from dataset.'''
    prompts = [" ".join(review.split()[:4]) for review in dataset['text']]
    return prompts

prompts = generate_prompts(imdb)
# %%
def generate_completion(prompt) -> str:
    '''
    Loads the GPT2-IMDB tokenizer and model, and generates completions for the given prompt (in the form of a string).

    Find name of model & tokenizer at the documentation page: https://huggingface.co/lvwerra/gpt2-imdb.

    Remember to set the `do_sample=True` flag when you call `model.generate`.
    '''
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = tokenizer.decode(model.generate(**inputs, do_sample=True, top_k=10, max_new_tokens=64).squeeze(0))
    return outputs

generate_completion(prompts[0])
# %%
def reward_model(samples, **kwargs) -> List[float]:
    '''
    Returns the rewards for the given samples (according to model which is defined inside function body).

    kwargs are passed to your model during a forward pass.
    '''
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    model = AutoModelForCausalLM.from_pretrained("lvwerra/distilbert-imdb")
    inputs = tokenizer(samples, padding=True, truncation=True, return_tensors='pt')
    rewards = []

    with t.inference_mode():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], **kwargs)

    logits = outputs.logits
    probabilities = t.softmax(logits, dim=1)

    for reward in probabilities:
       rewards.append(reward[1].item())

    return rewards

# %%
example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]
rewards = reward_model(example_strings)

tests.test_reward_model(rewards)
# %%
def create_pipeline(model_path):
    if t.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    # YOUR CODE HERE - Create a sentiment pipeline
    sentiment_fn = pipeline(
        "sentiment-analysis",
        model_path,
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    return sentiment_fn

sentiment_fn = create_pipeline("lvwerra/distilbert-imdb")
# %%
def get_positive_score(scores):
    '''
    Returns the score for the positive label.
    '''
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

def reward_model(samples: List[str], **kwargs) -> List[float]:
    '''
    Returns a list of reward values corresponding to the samples in `samples`.
    '''
    reward = list(map(get_positive_score, sentiment_fn(samples)))
    return reward
# %%
test_prompts = ['I am happy', 'I am sad']

rewards = reward_model(test_prompts)
tests.test_reward_test_prompts(rewards)

## Code below has an interesting set of examples:

print('I want to eat', reward_model('I want to eat'))
print('I want your puppy', reward_model('I want your puppy'))
print('I want to eat your puppy', reward_model('I want to eat your puppy'))
# %%
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
                max_new_tokens=100,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )
# %%
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
                top_k=10, # or can do top_p
                do_sample=True,
            ),
        ),
    )


def main() -> None:
    # Call the `train` function with appropriate arguments
    trainer = train(
        reward_fn = reward_model,
        prompts = prompts,
        eval_prompts = ['In my opinion'] * 256,
        config = ppo_config()
    )

gc.collect()
t.cuda.empty_cache()
main()
# %%
def main() -> None:
    # SOLUTION
    return train(
        reward_fn = reward_model,
        prompts = prompts,
        eval_prompts = ['I was extremely disappointed'] * 256, ## Feel free to try other negative prompts
        config =  ppo_config()
    )
# %%
def get_neutral_score(scores):
    return 1 - abs(dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"] - dict(map(lambda x: tuple(x.values()), scores))["NEGATIVE"])


def neutral_reward_model(samples: List[str], **kwargs) -> List[float]:
    # SOLUTION
    reward = list(map(get_neutral_score, sentiment_fn(samples)))
    return reward


def main() -> None:
    # SOLUTION
    trainer = train(
        reward_fn = neutral_reward_model,
        prompts = prompts,
        eval_prompts = ['In my opinion'] * 256, ## Feel free to try other negative prompts
        config =  ppo_config()
    )
# %%
