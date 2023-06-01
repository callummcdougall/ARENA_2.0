#%%
import torch as t
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import os
import sys

import plotly.express as px
import plotly.graph_objects as go

from functools import *
import gdown
from typing import List, Tuple, Union, Optional
from fancy_einsum import einsum
import einops
from jaxtyping import Float, Int
from tqdm import tqdm

from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_grokking_and_modular_arithmetic"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

root = (section_dir / 'Grokking' / 'saved_runs').resolve()
large_root = (section_dir / 'Grokking' / 'large_files').resolve()

from part5_grokking_and_modular_arithmetic.my_utils import *
import part5_grokking_and_modular_arithmetic.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"
# %%

if MAIN:
    p = 113

    cfg = HookedTransformerConfig(
        n_layers = 1,
        d_vocab = p+1,
        d_model = 128,
        d_mlp = 4 * 128,
        n_heads = 4,
        d_head = 128 // 4,
        n_ctx = 3,
        act_fn = "relu",
        normalization_type = None,
        device = device
    )

    model = HookedTransformer(cfg)
# %%

if MAIN:
    os.chdir(section_dir)
    if not large_root.exists(): 
        !git clone https://github.com/neelnanda-io/Grokking.git
        os.mkdir(large_root)

    full_run_data_path = (large_root / "full_run_data.pth").resolve()
    if not full_run_data_path.exists():
        url = "https://drive.google.com/uc?id=12pmgxpTHLDzSNMbMCuAMXP1lE_XiCQRy"
        output = str(full_run_data_path)
        gdown.download(url, output)
# %%

if MAIN:
    full_run_data = t.load(full_run_data_path)
    state_dict = full_run_data["state_dicts"][400]

    model = load_in_state_dict(model, state_dict)
# %%
if MAIN:
    lines(
        lines_list=[
            full_run_data['train_losses'][::10], 
            full_run_data['test_losses']
        ], 
        labels=['train loss', 'test loss'], 
        title='Grokking Training Curve', 
        x=np.arange(5000)*10,
        xaxis='Epoch',
        yaxis='Loss',
        log_y=True
    )
# %%

# Helper variables

if MAIN:
    W_O = model.W_O[0] #first dim is layers and we only have one
    W_K = model.W_K[0]
    W_Q = model.W_Q[0]
    W_V = model.W_V[0]
    W_in = model.W_in[0]
    W_out = model.W_out[0]
    W_pos = model.W_pos
    W_E = model.W_E[:-1]
    final_pos_resid_initial = model.W_E[-1] + W_pos[2]
    W_U = model.W_U[:, :-1]

    print('W_O  ', tuple(W_O.shape))
    print('W_K  ', tuple(W_K.shape))
    print('W_Q  ', tuple(W_Q.shape))
    print('W_V  ', tuple(W_V.shape))
    print('W_in ', tuple(W_in.shape))
    print('W_out', tuple(W_out.shape))
    print('W_pos', tuple(W_pos.shape))
    print('W_E  ', tuple(W_E.shape))
    print('W_U  ', tuple(W_U.shape))
# %%

if MAIN:
    all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(device)
    labels = t.tensor([fn(i, j) for i, j, _ in all_data]).to(device)
    original_logits, cache = model.run_with_cache(all_data)
    # cache is already an ActivationCache object
    # Final position only, also remove the logits for `=`
    # print(original_logits.shape)
    # original_logits has shape [12769, 3, 114], i.e., prediction of next token
    # for each of the 3 tokens in each of the 12769 combinations
    original_logits = original_logits[:, -1, :-1] # shape [12769, 113]
    original_loss = cross_entropy_high_precision(original_logits, labels)
    print(f"Original loss: {original_loss.item()}")
# %%

if MAIN:
    attn_mat = cache['blocks.0.attn.hook_pattern'][:,:,-1,:]
    neuron_acts_post = cache['blocks.0.mlp.hook_post'][:,-1,:]
    neuron_acts_pre = cache['blocks.0.mlp.hook_pre'][:,-1,:]
    

    # YOUR CODE HERE - get the relevant activations
    assert attn_mat.shape == (p*p, cfg.n_heads, 3)
    assert neuron_acts_post.shape == (p*p, cfg.d_mlp)
    assert neuron_acts_pre.shape == (p*p, cfg.d_mlp)
# %%
if MAIN:
    # YOUR CODE HERE - define these matrices
    W_logit = W_out @ W_U
    W_neur = W_E @ W_V @ W_O @ W_in
    t_2 = t.zeros(p+1).to(device)
    t_2[-1] = 1 #shape(p+1)
    embedded_t_2 = einops.einsum(t_2, model.W_E, 'p, p e -> e') # shape (e)
    W_QK = einops.einsum(W_Q, W_K, 'n e_Q h, n e_K h -> n e_Q e_K') # shape (n, e, e)
    W_attn = einops.einsum(embedded_t_2, W_QK, 'e, n e e_K -> n e_K')
    W_attn = einops.einsum(W_attn, W_E, 'n e, p e -> n p') // (cfg.d_head ** 0.5)

    assert W_logit.shape == (cfg.d_mlp, cfg.d_vocab - 1)
    assert W_neur.shape == (cfg.n_heads, cfg.d_vocab - 1, cfg.d_mlp)
    assert W_attn.shape == (cfg.n_heads, cfg.d_vocab - 1)

# %%
if MAIN:
    attn_mat = attn_mat[:, :, :2]
    # Note, we ignore attn from 2 -> 2

    attn_mat_sq = einops.rearrange(attn_mat, "(x y) head seq -> x y head seq", x=p)
    # We rearranged attn_mat, so the first two dims represent (x, y) in modular arithmetic equation

    inputs_heatmap(
        attn_mat_sq[..., 0], 
        title=f'Attention score for heads at position 0',
        animation_frame=2,
        animation_name='head'
    )
# %%
if MAIN:
    neuron_acts_post_sq = einops.rearrange(neuron_acts_post, "(x y) d_mlp -> x y d_mlp", x=p)
    neuron_acts_pre_sq = einops.rearrange(neuron_acts_pre, "(x y) d_mlp -> x y d_mlp", x=p)
    # We rearranged activations, so the first two dims represent (x, y) in modular arithmetic equation

    top_k = 3
    inputs_heatmap(
        neuron_acts_post_sq[..., :top_k], 
        title=f'Activations for first {top_k} neurons',
        animation_frame=2,
        animation_name='Neuron'
    )

# %%

if MAIN:
    top_k = 5
    animate_multi_lines(
        W_neur[..., :top_k], 
        y_index = [f'head {hi}' for hi in range(4)],
        labels = {'x':'Input token', 'value':'Contribution to neuron'},
        snapshot='Neuron',
        title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attention)'
    )
# %%

if MAIN:
    lines(
        W_attn,
        labels = [f'head {hi}' for hi in range(4)],
        xaxis='Input token',
        yaxis='Contribution to attn score',
        title=f'Contribution to attention score (pre-softmax) for each head'
    )
# %%

def make_fourier_basis(p: int) -> Tuple[Tensor, List[str]]:
    '''
    Returns a pair `fourier_basis, fourier_basis_names`, where `fourier_basis` is
    a `(p, p)` tensor whose rows are Fourier components and `fourier_basis_names`
    is a list of length `p` containing the names of the Fourier components (e.g. 
    `["const", "cos 1", "sin 1", ...]`). You may assume that `p` is odd.
    '''
    fourier_basis = t.ones((p,p)).to(device)
    fourier_basis_names = ['const']
    sin = 0
    cos = 0
    for i in range(1,p):
        if i % 2 == 1:
            sin +=1
            omega = 2*t.pi*sin/p
            fourier_basis[i] = t.sin(omega*t.arange(p))
            fourier_basis_names.append('sin '+ str(sin))
        else:
            cos +=1
            omega = 2*t.pi*cos/p
            fourier_basis[i] = t.cos(omega*t.arange(p))
            fourier_basis_names.append('cos '+ str(cos))
    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return (fourier_basis, fourier_basis_names)

#%%
# test requires python 3.10 or higher, didn't run but seems fine
if MAIN:
    tests.test_make_fourier_basis(make_fourier_basis)
# %%

if MAIN:
    fourier_basis, fourier_basis_names = make_fourier_basis(p)

    animate_lines(
        fourier_basis, 
        snapshot_index=fourier_basis_names, 
        snapshot='Fourier Component', 
        title='Graphs of Fourier Components (Use Slider)'
    )
# %%

if MAIN:
    imshow(fourier_basis @ fourier_basis.T)
# %%

def fft1d(x: t.Tensor) -> t.Tensor:
    '''
    Returns the 1D Fourier transform of `x`,
    which can be a vector or a batch of vectors.

    x.shape = (..., p)
    '''
    return einops.einsum(x, fourier_basis, '... p, q p -> ... q')

#%%
if MAIN:
    tests.test_fft1d(fft1d)
# %%

if MAIN:
    v = sum([
        fourier_basis[4],
        fourier_basis[15]/5,
        fourier_basis[67]/10
    ])

    line(v, xaxis='Vocab basis', title='Example periodic function')
    line(fft1d(v), xaxis='Fourier Basis', title='Fourier Transform of example function', hover=fourier_basis_names)
# %%

def fourier_2d_basis_term(i: int, j: int) -> Float[Tensor, "p p"]:
    '''
    Returns the 2D Fourier basis term corresponding to the outer product of the
    `i`-th component of the 1D Fourier basis in the `x` direction and the `j`-th
    component of the 1D Fourier basis in the `y` direction.

    Returns a 2D tensor of length `(p, p)`.
    '''
    return einops.einsum(fourier_basis[i,:], fourier_basis[j], 'p, q -> p q')

#%%
if MAIN:
    tests.test_fourier_2d_basis_term(fourier_2d_basis_term)
# %%
if MAIN:
    x_term = 4
    y_term = 6

    inputs_heatmap(
        fourier_2d_basis_term(x_term, y_term).T,
        title=f"2D Fourier Basis term {fourier_basis_names[x_term]}x {fourier_basis_names[y_term]}y"
    )
# %%

def fft2d(tensor: t.Tensor) -> t.Tensor:
    '''
    Retuns the components of `tensor` in the 2D Fourier basis.

    Asumes that the input has shape `(p, p, ...)`, where the
    last dimensions (if present) are the batch dims.
    Output has the same shape as the input.
    '''
    # out[i, j, ...] = fourier_2d_basis_term(i,j) * tensor[:,:,...]

    # dumb foorloopy way

    out = t.zeros(tensor.shape).to(device)
    for i in range(p):
        for j in range(p):
            out[i,j] = einops.einsum(fourier_2d_basis_term(i,j), tensor, 'p q, p q ... -> ...')
    correct_out = einops.einsum(tensor, fourier_basis, fourier_basis, 'px py ..., i px, j py -> i j ...')

    #assert t.all_close(out, correct_out)

    return out
#%%
if MAIN:
    tests.test_fft2d(fft2d)
# %%
if MAIN:
    example_fn = sum([
        fourier_2d_basis_term(4, 6), 
        fourier_2d_basis_term(14, 46) / 3,
        fourier_2d_basis_term(97, 100) / 6
    ])

    inputs_heatmap(example_fn.T, title=f"Example periodic function")
# %%

if MAIN:
    imshow_fourier(
        fft2d(example_fn),
        title='Example periodic function in 2D Fourier basis'
    )
# %%
inputs_heatmap(
    attn_mat[..., 0], 
    title=f'Attention score for heads at position 0',
    animation_frame=2,
    animation_name='head'
)
# %%
# Apply Fourier transformation

if MAIN:
    attn_mat_fourier_basis = fft2d(attn_mat_sq)

    # Plot results
    imshow_fourier(
        attn_mat_fourier_basis[..., 0], 
        title=f'Attention score for heads at position 0, in Fourier basis',
        animation_frame=2,
        animation_name='head'
    )
# %%
top_k = 3
inputs_heatmap(
    neuron_acts_post[:, :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
# %%
if MAIN:
    neuron_acts_post_fourier_basis = fft2d(neuron_acts_post_sq)

    top_k = 20
    imshow_fourier(
        neuron_acts_post_fourier_basis[..., :top_k], 
        title=f'Activations for first {top_k} neurons',
        animation_frame=2,
        animation_name='Neuron'
    )
# %%
top_k = 5
animate_multi_lines(
    W_neur[..., :top_k], 
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Input token', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attn)'
)
# %%
def fft1d_given_dim(tensor: t.Tensor, dim: int) -> t.Tensor:
    '''
    Performs 1D FFT along the given dimension (not necessarily the last one).
    '''
    return fft1d(tensor.transpose(dim, -1)).transpose(dim, -1)


if MAIN:
    W_neur_fourier = fft1d_given_dim(W_neur, dim=1)

    top_k = 5
    animate_multi_lines(
        W_neur_fourier[..., :top_k], 
        y_index = [f'head {hi}' for hi in range(4)],
        labels = {'x':'Fourier component', 'value':'Contribution to neuron'},
        snapshot='Neuron',
        hover=fourier_basis_names,
        title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attn), in Fourier basis'
    )
# %%
lines(
    W_attn,
    labels = [f'head {hi}' for hi in range(4)],
    xaxis='Input token',
    yaxis='Contribution to attn score',
    title=f'Contribution to attention score (pre-softmax) for each head'
)
# %%
if MAIN:
    lines(
        fft1d(W_attn), 
        labels = [f'head {hi}' for hi in range(4)],
        xaxis='Input token', 
        yaxis = 'Contribution to attn score',
        title=f'Contribution to attn score (pre-softmax) for each head, in Fourier Basis', 
        hover=fourier_basis_names
    )
# %%
