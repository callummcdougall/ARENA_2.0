# %%
import os
import time
import transformers
import torch
from transformer_lens import HookedTransformer

model_param_str = ['70m', '160m', '410m']
model_names = {mps: f"EleutherAI/pythia-{mps}-deduped" for mps in model_param_str}

# # GPU memory usage is 0 (as checked by nvidia-smi) after each model is loaded and garbage collected.
# for param_str, model_name in model_names.items():
#     model = transformers.GPTNeoXForCausalLM.from_pretrained(model_name)
#     model.cuda()

# del model
# import gc
# gc.collect()
# torch.cuda.empty_cache()

print(torch.cuda.memory_summary())
# Give the GPU some time to update its memory usage.
time.sleep(1)
os.system('nvidia-smi')

# GPU memory usage accumulates from past models.
# for param_str, model_name in model_names.items():
#     model = HookedTransformer.from_pretrained(model_name)

model= HookedTransformer.from_pretrained("gpt2-large")
assert model.mod_dict[""] == model

print(torch.cuda.memory_summary())
time.sleep(1)
os.system('nvidia-smi')

del model
import gc
gc.collect()
torch.cuda.empty_cache()

print(torch.cuda.memory_summary())
time.sleep(1)
os.system('nvidia-smi')
# %%
