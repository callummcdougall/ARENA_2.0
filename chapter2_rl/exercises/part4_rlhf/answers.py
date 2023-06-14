# %%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import json
import sys
import math
import gc
from pathlib import Path
import torch as t
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,  AutoModelForSequenceClassification, GenerationConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
import logging
from typing import cast, Any, List, Optional, Union, Tuple

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_rlhf"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import tests as tests
import utils as utils


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


your_text = "one two three one two three [MASK]"
predictions = predict(bert, tokenizer, your_text)
print("Model predicted: \n", "\n".join(map(str, predictions)))

# %%
imdb = load_dataset("imdb", split="train+test")

# %%
def label_split(dataset) -> Tuple[int, int]:
    n_pos = t.sum(t.Tensor(imdb['label']))
    print(f"Positive reviews: {n_pos}, Negative reviews: {len(imdb['label']) - n_pos}")
    return n_pos, len(imdb['label']) - n_pos

n_pos, n_neg = label_split(imdb)

tests.test_label_split(n_pos, n_neg)
# %%
prompts = []
def first_n_words(sample: str, n: int) -> str:
    return ' '.join(sample.split()[:n])

first_n_words('asdf wef w fw efw ef wef', 5)
prompts = [first_n_words(s, 5) for s in imdb['text']]
# %%
def generate_completion(prompt) -> str:
    '''
    Loads the GPT2-IMDB tokenizer and model, and generates completions for the given prompt (in the form of a string).

    Find name of model & tokenizer at the documentation page: https://huggingface.co/lvwerra/gpt2-imdb
    '''
    model = AutoModelForCausalLM.from_pretrained('lvwerra/gpt2-imdb')
    tokenizer = AutoTokenizer.from_pretrained('lvwerra/gpt2-imdb')
    inputs = tokenizer(prompt, return_tensors='pt')
    out = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_p=0.8).squeeze(0)
    return tokenizer.decode(out)

print(generate_completion(prompts[0]))
# %%
def reward_model(samples, **kwargs) -> List[float]:
    '''
    Returns the rewards for the given samples (according to model which is defined inside function body).

    kwargs are passed to your model during a forward pass.
    '''
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    inputs = tokenizer(samples, return_tensors='pt', padding=True)
    with t.inference_mode():
        outputs = model(**inputs).logits
    probs = t.softmax(outputs, dim=-1)[:, 1]
    return list(probs)

example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]
rewards = reward_model(example_strings)
# print([(example_strings[i], rewards[i]) for i in range(len(example_strings))])
tests.test_reward_model(rewards)
# %%

def create_pipeline(model_path):
    if t.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    # YOUR CODE HERE - Create a sentiment pipeline
    pipe = pipeline('sentiment-analysis', model=model_path, device=device, top_k=2, truncation=True, batch_size=256)
    return pipe

sentiment_fn = create_pipeline("lvwerra/distilbert-imdb")
# %%

def reward_model(samples: List[str], **kwargs) -> List[float]:
    '''
    Returns a list of reward values corresponding to the samples in `samples`.
    '''
    return [x[0]['score'] if x[0]['label'] == 'POSITIVE' else x[1]['score'] for x in sentiment_fn(samples)]

reward_model(['test string', 'another string'])
# %%
test_prompts = ['I am happy', 'I am sad']

rewards = reward_model(test_prompts)
tests.test_reward_test_prompts(rewards)

## Code below has an interesting set of examples:

print('I want to eat', reward_model('I want to eat'))
print('I want your puppy', reward_model('I want your puppy'))
print('I want to eat your puppy', reward_model('I want to eat your puppy'))
print('I would love to eat your puppy', reward_model('I would love to eat your puppy'))

# %%
