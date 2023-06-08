# %%
import torch
import gc
import pdb
from pympler import  refbrowser
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
del model
gc.collect()
torch.cuda.empty_cache()

first_obj = None

count = 0
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(count, type(obj), obj.size())
            first_obj = obj
            break
                
    except Exception as e:
        pass

refbrowser.ConsoleBrowser(first_obj, str_func=lambda x: pdb.set_trace() or str(x)).print_tree()


print(f"Found {count} dangling objects")

"""
torch.nn.parameter.Parameter-+-dict-+-module(__main__)-+-dict
                             |      |                  +-dict
                             |      |                  +-dict
                             |      |                  +-dict
                             |      |                  +-list
                             |      |
                             |      +-list--list_iterator
                             |
                             +-list--list_iterator
"""