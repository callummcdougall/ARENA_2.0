# %%
import os
import time
import transformers
import torch
import gc

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

# print(torch.cuda.memory_summary())
# # Give the GPU some time to update its memory usage.
# time.sleep(1)
# os.system('nvidia-smi')

# GPU memory usage accumulates from past models.
# for param_str, model_name in model_names.items():
#     model = HookedTransformer.from_pretrained(model_name)

from pympler import tracker, summary, refbrowser
import pdb 

t = tracker.SummaryTracker()

# t.print_diff()
from transformer_lens import HookedTransformer

# time.sleep(1)
# os.system("nvidia-smi")

model = HookedTransformer.from_pretrained("gpt2-small")
del model

gc.collect()
torch.cuda.empty_cache()
i = 0

objs = []

for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            i += 1
            if i < 5:
                print(type(obj), obj.size())
                # cb = refbrowser.ConsoleBrowser(obj)
                # cb.print_tree()
                
    except Exception as e:
        # print(str(e))
        pass

# cb = refbrowser.ConsoleBrowser(objs[0])
# cb.print_tree()

# cb = refbrowser.ConsoleBrowser(objs[1])
# cb.print_tree()


# time.sleep(1)
# os.system("nvidia-smi")
# cb = refbrowser.ConsoleBrowser(li[0])
# # cb = refbrowser.ConsoleBrowser(li[0], str_func= lambda x: pdb.set_trace() or str(type(x)))
# cb.print_tree()



# t.print_diff()

# torch.cuda.empty_cache()

# t.print_diff()

# diff1 = summary.get_diff(t.s0, t.s1)
# diff2 = summary.get_diff(t.s1, t.s2)
# pass
# print(torch.cuda.memory_summary())
# time.sleep(1)
# os.system('nvidia-smi')
# %%
if __name__ == '__main__':
    print(1)
