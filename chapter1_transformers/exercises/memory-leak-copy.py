# %%
from transformer_lens import HookedTransformer
from pympler import refbrowser



model, init_model = HookedTransformer.from_pretrained("gpt2-small")
# assert init_model.mod_dict[""] == init_model
# %%
embed = init_model.W_E

cb = refbrowser.ConsoleBrowser(embed)
cb.print_tree()

# %%

dict(model.named_parameters()).keys()