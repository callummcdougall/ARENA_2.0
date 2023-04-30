import os
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

st_image("leaves.png", 350)
st.markdown("Coming soon!")





# if MAIN:
#     clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
#     corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

#     clean_tokens = gpt2_small.to_tokens(clean_prompt)
#     corrupted_tokens = gpt2_small.to_tokens(corrupted_prompt)

# def logits_to_logit_diff(logits, correct_answer=" John", incorrect_answer=" Mary"):
#     # model.to_single_token maps a string value of a single token to the token index for that token
#     # If the string is not a single token, it raises an error.
#     correct_index = gpt2_small.to_single_token(correct_answer)
#     incorrect_index = gpt2_small.to_single_token(incorrect_answer)
#     return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

# if MAIN:
#     # We run on the clean prompt with the cache so we store activations to patch in later.
#     clean_logits, clean_cache = gpt2_small.run_with_cache(clean_tokens, remove_batch_dim=True)
#     clean_logit_diff = logits_to_logit_diff(clean_logits)
#     print(f"Clean logit difference: {clean_logit_diff.item():.3f}")

#     # We don't need to cache on the corrupted prompt.
#     corrupted_logits = gpt2_small(corrupted_tokens)
#     corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
#     print(f"Corrupted logit difference: {corrupted_logit_diff.item():.3f}")

# # %%

# # We define a residual stream patching hook
# # We choose to act on the residual stream at the start of the layer, so we call it resid_pre
# # The type annotations are a guide to the reader and are not necessary
# def residual_stream_patching_hook(
#     resid_pre: TT["batch", "pos", "d_model"],
#     hook: HookPoint,
#     position: int,
#     clean_cache: ActivationCache
# ) -> TT["batch", "pos", "d_model"]:
#     # Each HookPoint has a name attribute giving the name of the hook.
#     clean_resid_pre = clean_cache[hook.name]
#     resid_pre[:, position, :] = clean_resid_pre[position, :]
#     return resid_pre

# if MAIN:
#     # We make a tensor to store the results for each patching run. We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
#     num_positions = len(clean_tokens[0])
#     ioi_patching_result = t.zeros((gpt2_small.cfg.n_layers, num_positions), device=gpt2_small.cfg.device)
#     gpt2_small.reset_hooks()

#     for layer in tqdm(range(gpt2_small.cfg.n_layers)):
#         for position in range(num_positions):
#             # Use functools.partial to create a temporary hook function with the position fixed
#             temp_hook_fn = functools.partial(residual_stream_patching_hook, position=position, clean_cache=clean_cache)
#             # Run the model with the patching hook
#             patched_logits = gpt2_small.run_with_hooks(corrupted_tokens, fwd_hooks=[
#                 (utils.get_act_name("resid_pre", layer), temp_hook_fn)
#             ])
#             # Calculate the logit difference
#             patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
#             # Store the result, normalizing by the clean and corrupted logit difference so it's between 0 and 1 (ish)
#             ioi_patching_result[layer, position] = (patched_logit_diff - corrupted_logit_diff)/(clean_logit_diff - corrupted_logit_diff)

# # %%

# if MAIN:
#     # Add the index to the end of the label, because plotly doesn't like duplicate labels
#     token_labels = [f"{token}_{index}" for index, token in enumerate(gpt2_small.to_str_tokens(clean_tokens))]
#     imshow(ioi_patching_result, x=token_labels, xaxis="Position", yaxis="Layer", title="Normalized Logit Difference After Patching Residual Stream on the IOI Task")
# %%



