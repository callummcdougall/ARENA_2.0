### IMPORTS
import einops
from functools import partial
import numpy as np
import plotly.express as px
import torch as t
from transformer_lens import utils

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
t.set_grad_enabled(False)

### PLOTTING FUNCTIONS
# %%
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def two_lines(tensor1, tensor2, renderer=None, **kwargs):
    px.line(y=[utils.to_numpy(tensor1), utils.to_numpy(tensor2)], **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


### INTERPRETABILITY HELPERS

### Get logit diffs for different outputs
def logit_diff(logits, answer_tokens, per_prompt=False):
    # We only take the final logits
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()

def get_prompt_and_answer_tokens(
    model,
    prompts=None,
    answers=None
):
    if prompts is None:
        prompts = [
            "This movie was really",
        ]
    if answers is None:
        answers = [
            (" bad", " good"),
        ]

    prompt_tokens = model.to_tokens(prompts, prepend_bos=True)
    
    # List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
    answer_tokens = []
    for i in range(len(prompts)):
        answer_tokens.append(
            (
                model.to_single_token(answers[i][0]),
                model.to_single_token(answers[i][1]),
            )
        )
    answer_tokens = t.tensor(answer_tokens).to(device)

    return prompt_tokens, answer_tokens

# Logit Lens
def residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions, n_prompts):
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
    return einops.einsum(scaled_residual_stack, logit_diff_directions, "... batch d_model, batch d_model -> ...")/n_prompts

# Residual Stream Attributions
def plot_logits_diffs_at_residual_stream(base_cache, rlhf_cache, logit_diff_directions, n_prompts, n_layers):
    accumulated_residual, labels = base_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
    logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, rlhf_cache, logit_diff_directions, n_prompts)

    accumulated_residual_rlhf, labels = rlhf_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
    logit_lens_logit_diffs_rlhf = residual_stack_to_logit_diff(accumulated_residual_rlhf, rlhf_cache, logit_diff_directions, n_prompts)

    two_lines(logit_lens_logit_diffs, logit_lens_logit_diffs_rlhf, x=np.arange(n_layers*2+1)/2, hover_name=labels, title="Logit Difference From Accumulated Residual Stream")

# Layer Attribution
def plot_logit_diffs_for_all_layers(base_cache, rlhf_cache, logit_diff_directions, n_prompts):
    per_layer_residual, labels = base_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, base_cache, logit_diff_directions, n_prompts)

    per_layer_residual_rlhf, labels = rlhf_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs_rlhf = residual_stack_to_logit_diff(per_layer_residual_rlhf, rlhf_cache, logit_diff_directions, n_prompts)

    two_lines(per_layer_logit_diffs, per_layer_logit_diffs_rlhf, hover_name=labels, title="Logit Difference From Each Layer")


# PATCHING FUNCTIONS
# We will use this function to patch different parts of the residual stream
def patch_residual_component(
    to_residual_component,
    hook,
    subcomponent_index, 
    from_cache):
    from_cache_component = from_cache[hook.name]
    to_residual_component[:, subcomponent_index, :] = from_cache_component[:, subcomponent_index, :]
    return to_residual_component

# We will use this to patch specific heads
def patch_head_vector(
    rlhf_head_vector,
    hook, 
    subcomponent_index, 
    from_cache):
    if isinstance(subcomponent_index, int):
      rlhf_head_vector[:, :, subcomponent_index, :] = from_cache[hook.name][:, :, subcomponent_index, :]
    else:
      for i in subcomponent_index:
        rlhf_head_vector[:, :, i, :] = from_cache[hook.name][:, :, i, :]
    return rlhf_head_vector

# We will use this to patch specific heads
def normalize_patched_logit_diff(patched_logit_diff, original_average_logit_diff_rlhf, original_average_logit_diff_source):
    # Subtract corrupted logit diff to measure the improvement, divide by the total improvement from clean to corrupted to normalize
    # 0 means zero change, negative means more positive, 1 means equivalent to RLHF model, >1 means more negative than RLHF model
    return (patched_logit_diff - original_average_logit_diff_source)/(original_average_logit_diff_rlhf - original_average_logit_diff_source)



# Patch Residual Stream
def run_rs_patching_experiments(
        model, 
        metric, 
        prompt_tokens,
        patching_fn=patch_residual_component,
        patching_cache=None,
        activations_to_patch="resid_pre", 
        normalization_fn=None,
        title="Logit Difference From Patched Residual Stream"
):
    results = t.zeros(model.cfg.n_layers, prompt_tokens.shape[1], device=device, dtype=t.float32)

    for layer in range(model.cfg.n_layers):
        for position in range(prompt_tokens.shape[1]):
                hook_fn = partial(patching_fn, subcomponent_index=position, from_cache=patching_cache)
                patched_logits = model.run_with_hooks(
                    prompt_tokens, 
                    fwd_hooks = [(utils.get_act_name(activations_to_patch, layer), 
                        hook_fn)], 
                    return_type="logits"
                )
                patching_results = metric(patched_logits)

                if normalization_fn is not None:
                    results[layer, position] = normalization_fn(patching_results)
                else:
                    results[layer, position] = patching_results

    prompt_position_labels = [f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(prompt_tokens[0]))]
    imshow(results, x=prompt_position_labels, title=title, xaxis="Position", yaxis="Layer")

def run_head_patching_experiments(
        model, 
        metric, 
        prompt_tokens,
        patching_fn=patch_residual_component,
        patching_cache=None,
        activations_to_patch="resid_pre", 
        normalization_fn=None,
        title="Logit Difference From Patched Residual Stream"
):
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

    for layer in range(model.cfg.n_layers):
        for head_index in range(model.cfg.n_heads):
                hook_fn = partial(patching_fn, subcomponent_index=head_index, from_cache=patching_cache)
                patched_logits = model.run_with_hooks(
                    prompt_tokens, 
                    fwd_hooks = [(utils.get_act_name(activations_to_patch, layer), 
                        hook_fn)], 
                    return_type="logits"
                )
                patching_results = metric(patched_logits)

                if normalization_fn is not None:
                    results[layer, head_index] = normalization_fn(patching_results)
                else:
                    results[layer, head_index] = patching_results

    imshow(results, title=title, xaxis="Position", yaxis="Layer")

### COMPARE MODELS
# %%
def compare_model_generations_with_answer(prompt, answer, source_model, rlhf_model):
    utils.test_prompt(prompt, answer, source_model, prepend_bos=True)
    utils.test_prompt(prompt, answer, rlhf_model, prepend_bos=True)

# %%
def get_rlhf_model_logit_diffs_for_answers(
        rhlf_model, 
        answer_tokens, 
        source_cache, 
        rlhf_cache,
        original_average_logit_diff_source, 
        original_average_logit_diff_rlhf,
        n_prompts
    ):
    # Here we get the unembedding vectors for the answer tokens
    answer_residual_directions = rhlf_model.tokens_to_residual_directions(answer_tokens)
    print("Answer residual directions shape:", answer_residual_directions.shape)

    logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    print("Logit difference directions shape:", logit_diff_directions.shape)
    # Cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 
    final_residual_stream_source = source_cache["resid_post", -1]
    final_residual_stream_rlhf = rlhf_cache["resid_post", -1]
    print("Final residual stream shape:", final_residual_stream_rlhf.shape)
    final_token_residual_stream_source = final_residual_stream_source[:, -1, :]
    final_token_residual_stream_rlhf = final_residual_stream_rlhf[:, -1, :]

    # Apply LayerNorm scaling
    # pos_slice is the subset of the positions we take - here the final token of each prompt
    scaled_final_token_residual_stream_source = source_cache.apply_ln_to_stack(final_token_residual_stream_source, layer = -1, pos_slice=-1)
    scaled_final_token_residual_stream_rlhf = rlhf_cache.apply_ln_to_stack(final_token_residual_stream_rlhf, layer = -1, pos_slice=-1)
    # %%

    print("\nSource Model:")
    average_logit_diff = einops.einsum(scaled_final_token_residual_stream_source, logit_diff_directions, "batch d_model, batch d_model -> ")/n_prompts
    print("Calculated scaled average logit diff:", average_logit_diff.item())
    print("Original logit difference:",original_average_logit_diff_source.item())

    print("\nRLHF Model:")
    average_logit_diff = einops.einsum(scaled_final_token_residual_stream_rlhf, logit_diff_directions, "batch d_model, batch d_model -> ")/n_prompts
    print("Calculated scaled average logit diff:", average_logit_diff.item())
    print("Original logit difference:",original_average_logit_diff_rlhf.item())

    return logit_diff_directions

