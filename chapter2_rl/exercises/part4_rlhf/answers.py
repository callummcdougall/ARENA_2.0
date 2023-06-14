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