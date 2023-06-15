#%%
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
import wandb
# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_rlhf"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part4_rlhf.tests as tests
import part4_rlhf.utils as utils
from part4_rlhf.trlx.trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from part4_rlhf.trlx.trlx.models.modeling_ppo import PPOConfig
from part4_rlhf.trlx.trlx import train

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
#%%
from functools import reduce
def label_split(dataset) -> Tuple[List[str], List[str]]:
    # n_pos = []
    # n_neg = []
    # any(n_pos.append(text) for text, label in dataset if label == 0 or n_neg.append(text))
    # any((n_pos if int(label) == 1 else n_neg).append(text) for text, label in dataset)
    # return reduce(lambda acc, row: (acc[0].append(row[0]) or acc) if row[1] == 1 else (acc[1].append(row[0]) or acc), dataset, ([], []))
    return dataset['label'].count(1), dataset['label'].count(0)

n_pos, n_neg = label_split(imdb)

tests.test_label_split(n_pos, n_neg)
#%%
def generate_prompts(dataset):
    prompts = [" ".join(review.split()[:4]) for review in dataset["text"]]
    return prompts

prompts = generate_prompts(imdb)# YOUR CODE HERE - fill in the prompts
# %%
tokenizer_gpt = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
model_gpt = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
def generate_completion(prompt) -> str:
    '''
    Loads the GPT2-IMDB tokenizer and model, and generates completions for the given prompt (in the form of a string).
    '''
    tokens = tokenizer_gpt(prompt, return_tensors='pt')
    completion = model_gpt.generate(**tokens, do_sample=True, top_k=10, max_new_tokens=64).squeeze(0)
    return tokenizer_gpt.decode(completion)

generate_completion(prompts[0])
# %%
# reward_model_bruh = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
# reward_model_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
# reward_model_bruh.eval()
# def reward_model(samples, **kwargs):
#     '''
#     Returns the rewards for the given samples, using the reward model `model`.

#     kwargs are passed to your model during a forward pass.
#     '''
#     return reward_model_bruh(**reward_model_tokenizer(samples, return_tensors='pt', padding=True), **kwargs)



# example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]
# rewards = reward_model(example_strings)
# tests.test_reward_model(rewards)
# %%
def create_pipeline(model_path):
    if t.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    return pipeline('sentiment-analysis', model=model_path, top_k=2,
                    truncation=True, batch_size=256, device=device)

# sentiment_fn = create_pipeline("lvwerra/distilbert-imdb")
# # %%
# def reward_model(samples: List[str], **kwargs) -> List[float]:
#     '''
#     Returns a list of reward values corresponding to the samples in `samples`.
#     '''
#     return [reduce(lambda s, d: d['score'] if d['label'] == 'POSITIVE' else s, row, 0) for row in sentiment_fn(samples, )]
# # %%
# test_prompts = ['I am happy', 'I am sad']

# rewards = reward_model(test_prompts)
# tests.test_reward_test_prompts(rewards)

## Code below has an interesting set of examples:

# print('I want to eat', reward_model('I want to eat'))
# print('I want your puppy', reward_model('I want your puppy'))
# print('I want to eat your puppy', reward_model('I want to eat your puppy'))
# %%
def ppo_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=1000,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="lvwerra/gpt2-imdb", truncation_side="right"),
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
def main() -> None:
    # Call the `train` function with appropriate arguments
    train(
        reward_fn=reward_model,
        prompts=prompts,
        eval_prompts=['Never gonna give you up', 'Never gonna let you down', 'Never gonna run around and desert you', 'Never gonna make you cry', 'Never gonna say goodbye', 'Never gonna tell a lie, and hurt you'],
        config=ppo_config()
    )

gc.collect()
t.cuda.empty_cache()
# main()
# %%



#! AAAAAAAAA
# n=int(input())
# list=[2, 24, 91963648, 10200236032, 4320, 4680, 26208, 20427264, 197064960, 21857648640, 57575890944,88898072401645056, 301183421949935616, 8910720, 17428320, 8583644160, 57629644800, 206166804480, 1416963251404800, 15338300494970880, 6275163455171297280, 200286975596707184640, 215594611071909888000, 5997579964837140234240, 39887491844324122951680, 17116004505600, 75462255348480000, 6219051710415667200, 14031414189615513600, 352444116692828160000, 835095457414213632000, 59485231752222033838080, 64031599488357236736000]
# ans=0
# for i in list:
#     if i<=n:
#         ans+=i
# print(ans)

# print(sum((lambda n: [x for x in [2, 24, 91963648, 10200236032, 4320, 4680, 26208, 20427264, 197064960, 21857648640, 57575890944,88898072401645056, 301183421949935616, 8910720, 17428320, 8583644160, 57629644800, 206166804480, 1416963251404800, 15338300494970880, 6275163455171297280, 200286975596707184640, 215594611071909888000, 5997579964837140234240, 39887491844324122951680, 17116004505600, 75462255348480000, 6219051710415667200, 14031414189615513600, 352444116692828160000, 835095457414213632000, 59485231752222033838080, 64031599488357236736000] if x <= n])(int(input()))))
# %%
rickroll = """
We're no strangers to love
You know the rules and so do I (do I)
A full commitment's what I'm thinking of
You wouldn't get this from any other guy
I just wanna tell you how I'm feeling
Gotta make you understand
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
We've known each other for so long
Your heart's been aching, but you're too shy to say it (say it)
Inside, we both know what's been going on (going on)
We know the game and we're gonna play it
And if you ask me how I'm feeling
Don't tell me you're too blind to see
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
We've known each other for so long
Your heart's been aching, but you're too shy to say it (to say it)
Inside, we both know what's been going on (going on)
We know the game and we're gonna play it
I just wanna tell you how I'm feeling
Gotta make you understand
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
"""
#%%
def bincount(tokens):
    return t.clamp(t.where(t.arange(50257) == tokenizer_gpt.eos_token_id, t.tensor(0), t.stack([t.bincount(tokens['input_ids'][i], minlength=50257) for i in range(len(tokens['input_ids']))])), 10e-7, len(tokens['input_ids']))

rickroll_tokens = tokenizer_gpt(rickroll, return_tensors='pt')
rickroll_bincounts = bincount(rickroll_tokens)
rickroll_bincounts = rickroll_bincounts / rickroll_bincounts.sum(-1, keepdim=True)

#%%
# fluency = create_pipeline("dennlinger/roberta-cls-consec")
def lcs(X , Y):
    X = X.split()
    Y = Y.split()
    M = len(X)
    N = len(Y)

    LCSuff = [[0 for k in range(N+1)] for l in range(M+1)]
    mx = 0
    for i in range(M + 1):
        for j in range(N + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                mx = max(mx, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return mx


tokenizer_gpt.pad_token = tokenizer_gpt.eos_token

#%%
def reward_model(samples: List[str], **kwargs) -> List[float]:
    # rewards = []
    # for sample in samples:
    #     reward = 0
    #     words = ' '.join(sample.split("\n")).split(' ')
    #     for i, word in enumerate(words):
    #         if word in rickroll:
    #             reward += 0.1
    #         if word == rickroll[i]:
    #             reward += 1
    #     rewards.append(reward)
    # return rewards


    # tokens = tokenizer_gpt(samples, return_tensors='pt')
    # resid = model_gpt(**tokens, return_dict=True).last_hidden_state

    # fluency_rewards = fluency(samples)

    # return [fluency_rewards[i]['score'] + t.cosine_similarity(rickroll_resid, resid[i].unsqueeze(0)).item() for i in range(len(samples))]

    # reward 1: largest common subsequence
    lcs_reward = t.tensor([lcs(sample, rickroll) for sample in samples]).pow(2)

    # reward 2: distribution of words in samples is similar to distribution of words in rickroll
    sample_tokens = tokenizer_gpt(samples, return_tensors='pt', padding=True)
    sample_bincounts = bincount(sample_tokens)
    sample_bincounts = sample_bincounts / sample_bincounts.sum(-1, keepdim=True)
    # compare the distributions of the two bincounts
    avg_bincount = (sample_bincounts + rickroll_bincounts) / 2
    kl1 = t.nn.functional.kl_div(sample_bincounts.log(), avg_bincount, reduction='none').sum(-1)
    kl2 = t.nn.functional.kl_div(rickroll_bincounts.log(), avg_bincount, reduction='none').sum(-1)
    kl_divergence = (kl1 + kl2) / 2
    sample_lengths = t.where(sample_tokens['input_ids'] == tokenizer_gpt.eos_token_id, t.tensor(0), t.tensor(1)).sum(-1)
    kl_divergence = kl_divergence * sample_lengths / 10
    kl_reward = 100 / (kl_divergence + 1e-3)

    wandb.log({'lcs_reward': lcs_reward,
               'lcs_reward_mean': lcs_reward.mean(dtype=t.float32),
               'kl_reward': kl_reward,
               'kl_reward_mean': kl_reward.mean(dtype=t.float32)})

    return (lcs_reward + kl_reward)/100
    
    # reward 3: fluency (maybe add this later)
        
def main() -> None:
    # Call the `train` function with appropriate arguments
    train(
        reward_fn=reward_model,
        prompts=prompts,
        eval_prompts=['This is such a good', 'I found it really interesting how', 'Why are you gay?'],
        config=ppo_config()
    )

gc.collect()
t.cuda.empty_cache()
main()
# %%
