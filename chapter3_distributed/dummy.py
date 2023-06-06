# %%
import os
import gc;

import tqdm

# gc.set_debug(gc.DEBUG_STATS)
import time
# import transformers
# os.system("pip install pympler")
# from pympler import  refbrowser
import torch
torch.cuda.set_per_process_memory_fraction(0.5, 0)
from transformer_lens import HookedTransformer

# model_param_str = ['70m', '160m', '410m']
# model_names = {mps: f"EleutherAI/pythia-{mps}-deduped" for mps in model_param_str}

# # # GPU memory usage is 0 (as checked by nvidia-smi) after each model is loaded and garbage collected.
# for param_str, model_name in model_names.items():
#     model = transformers.GPTNeoXForCausalLM.from_pretrained(model_name)
#     model.cuda()

# del model
# import gc
# gc.collect()
# torch.cuda.empty_cache()

# print(torch.cuda.memory_summary())
# Give the GPU some time to update its memory usage.
# time.sleep(1)
# os.system('nvidia-smi')

# GPU memory usage accumulates from past models.
# for param_str, model_name in model_names.items():
#     model = HookedTransformer.from_pretrained(model_name)

# del model
print("Loading model")
# start = list(o for o in gc.get_objects())
model = HookedTransformer.from_pretrained("gpt2-small")
gc.collect()
gc.collect()
torch.cuda.empty_cache()
print("===================================== peak usage")
os.system('nvidia-smi')
del model
gc.collect()
gc.collect()
torch.cuda.empty_cache()
t = torch.zeros((2000,1000,1000), device='cuda', dtype=torch.uint8)
print("===================================== after creating tensor")
os.system('nvidia-smi')
del t
gc.collect()
gc.collect()
torch.cuda.empty_cache()
print("===================================== after deleting model and tensor")
os.system('nvidia-smi')

# end = list(o for o in gc.get_objects())
# print("Loaded model")
# start_ids = [id(o) for o in start]
# diff = [o for o in tqdm.tqdm(end) if id(o) not in start_ids]
# print(len(diff))
print("deleting model")
print("deleted model")

# import gc
gc.collect()
torch.cuda.empty_cache()


# import pdb
# from pympler import refbrowser
# cb = refbrowser.ConsoleBrowser(model.W_E, str_func=lambda x: pdb.set_trace() or str(type(x)) + "=" * 25)
# cb.print_tree()

# assert model.mod_dict[""] == model

# # print(torch.cuda.memory_summary())
# # time.sleep(1)
# # os.system('nvidia-smi')

# del HookedTransformer
# print(torch.cuda.memory_summary())
# time.sleep(1)
# os.system('nvidia-smi')

# def fn(x):
#     pdb.set_trace()
#     return str(type(x))
# objs = []
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             objs.append(obj)
#     except Exception as e:
#         pass
# print(len(objs))
#
# del objs
# gc.collect()

objs = []
for obj in gc.get_objects():
    try:
        if 'transf' in str(obj.__class__).lower():
            objs.append(obj)
    except Exception as e:
        pass

# l = len(objs)
print(len(objs))
# for i in range(10,l):
#     print(i)
#     try:
#         cb = refbrowser.ConsoleBrowser(objs[i], )
#         cb.print_tree()
#     except:
#         pass
# print(torch.cuda.memory_summary())
# time.sleep(1)
# os.system('nvidia-smi')
# %%