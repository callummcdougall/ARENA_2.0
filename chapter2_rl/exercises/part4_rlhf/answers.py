# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import json
import sys
import math
import gc
from pathlib import Path
import torch as t
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.models.bert.modeling_bert import BertForMaskedLM
import logging
from typing import cast, Any, List, Optional, Union, Tuple

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_rlhf"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part4_rlhf.tests as tests
import part4_rlhf.utils as utils
from part4_rlhf.trlx.trlx.data.default_configs import (
    TRLConfig,
    TrainConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    ModelConfig,
)
from part4_rlhf.trlx.trlx.models.modeling_ppo import PPOConfig
from part4_rlhf.trlx.trlx import train

# %%
bert = utils.load_pretrained_bert()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def predict(
    model: BertForMaskedLM, tokenizer: AutoTokenizer, text: str, k=15
) -> List[List[str]]:
    """
    Return a list of k strings for each [MASK] in the input.
    """

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
def label_split(dataset) -> None:
    pass


n_pos, n_neg = label_split(imdb)

tests.test_label_split(n_pos, n_neg)
# %%

prompts = [
    "The movie was",
    "The worst thing about the movie was",
    "On the weekend I look forward to",
    "In hindsight, we should never have",
    "Oh my god the",
    "It all became clear to me when",
    "It's the ",
    "I should have realized that",
]


# YOUR CODE HERE - fill in the prompts
# %%
