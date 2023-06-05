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
# if MAIN:
#     tests.test_make_fourier_basis(make_fourier_basis)
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
# if MAIN:
#     tests.test_fft1d(fft1d)
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
# if MAIN:
#     tests.test_fourier_2d_basis_term(fourier_2d_basis_term)
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
# if MAIN:
#     tests.test_fft2d(fft2d)
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

# SECTION 2

# fourier_basis @ W_E gives a linear combination of fourier basis vectors for each item in vocab?
if MAIN:
    line(
        (fourier_basis @ W_E).pow(2).sum(1), 
        hover=fourier_basis_names,
        title='Norm of embedding of each Fourier Component',
        xaxis='Fourier Component',
        yaxis='Norm'
    )
# %%

top_k = 5
inputs_heatmap(
    neuron_acts_post_sq[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
imshow_fourier(
    neuron_acts_post_fourier_basis[..., :top_k], 
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
# %%
if MAIN:
    neuron_acts_centered = neuron_acts_post_sq - neuron_acts_post_sq.mean(dim=(0,1), keepdims=True)
    neuron_acts_centered_fourier = fft2d(neuron_acts_centered)
    imshow_fourier(
        neuron_acts_centered_fourier.pow(2).mean(-1),
        title=f"Norms of 2D Fourier components of centered neuron activations",
    )
# %%
def arrange_by_2d_freqs(tensor):
    '''
    Takes a tensor of shape (p, p, ...) and returns a tensor of shape
    (p//2 - 1, 3, 3, ...) representing the Fourier coefficients sorted by
    frequency (each slice contains const, linear and quadratic terms).

    In other words, if the first two dimensions of the original tensor
    correspond to indexing by 2D Fourier frequencies as follows:

        1           cos(w_1*x)            sin(w_1*x)           ...
        cos(w_1*y)  cos(w_1*x)cos(w_1*y)  sin(w_1*x)cos(w_1*y) ...
        sin(w_1*y)  cos(w_1*x)sin(w_1*y)  sin(w_1*x)sin(w_1*y) ...
        cos(w_2*y)  cos(w_1*x)cos(w_2*y)  sin(w_1*x)cos(w_2*y) ...
        ...

    Then the (k-1)-th slice of the new tensor are the terms corresponding to 
    the following 2D Fourier frequencies:

        1           cos(w_k*x)            sin(w_k*x)           ...
        cos(w_k*y)  cos(w_k*x)cos(w_k*y)  sin(w_k*x)cos(w_k*y) ...
        sin(w_k*y)  cos(w_k*x)sin(w_k*y)  sin(w_k*x)sin(w_k*y) ...

    for k = 1, 2, ..., p//2.

    Note we omit the constant term, i.e. the 0th slice has frequency k=1.
    '''
    idx_2d_y_all = []
    idx_2d_x_all = []
    for freq in range(1, p//2):
        idx_1d = [0, 2*freq-1, 2*freq]
        idx_2d_x_all.append([idx_1d for _ in range(3)])
        idx_2d_y_all.append([[i]*3 for i in idx_1d])
    return tensor[idx_2d_y_all, idx_2d_x_all]


def find_neuron_freqs(
    fourier_neuron_acts: Float[Tensor, "p p d_mlp"]
) -> Tuple[Float[Tensor, "d_mlp"], Float[Tensor, "d_mlp"]]:
    '''
    Returns the tensors `neuron_freqs` and `neuron_frac_explained`, 
    containing the frequencies that explain the most variance of each 
    neuron and the fraction of variance explained, respectively.
    '''
    fourier_neuron_acts_by_freq = arrange_by_2d_freqs(fourier_neuron_acts)
    assert fourier_neuron_acts_by_freq.shape == (p//2-1, 3, 3, d_mlp)

    sum_of_squares = einops.einsum(fourier_neuron_acts_by_freq, fourier_neuron_acts_by_freq, 'q a b d_mlp, q a b d_mlp -> q d_mlp')

    neuron_freqs = t.argmax(sum_of_squares, dim=0)+1

    # sum of squares of all fourier coefficients for this neuron
    total_variance = einops.einsum(fourier_neuron_acts, fourier_neuron_acts, 'x y neuron, x y neuron -> neuron')

    # variance of neuron's highest activation / total variance
    neuron_frac_explained = sum_of_squares.max(0)[0] / total_variance

    return neuron_freqs, neuron_frac_explained


neuron_freqs, neuron_frac_explained = find_neuron_freqs(neuron_acts_centered_fourier)
key_freqs, neuron_freq_counts = t.unique(neuron_freqs, return_counts=True)

assert key_freqs.tolist() == [14, 35, 41, 42, 52]
# %%

fraction_of_activations_positive_at_posn2 = (cache['pre', 0][:, -1] > 0).float().mean(0)

scatter(
    x=neuron_freqs, 
    y=neuron_frac_explained,
    xaxis="Neuron frequency", 
    yaxis="Frac explained", 
    colorbar_title="Frac positive",
    title="Fraction of neuron activations explained by key freq",
    color=utils.to_numpy(fraction_of_activations_positive_at_posn2)
)
# %%

# To represent that they are in a special sixth cluster, we set the frequency of these neurons to -1
neuron_freqs[neuron_frac_explained < 0.85] = -1.
key_freqs_plus = t.concatenate([key_freqs, -key_freqs.new_ones((1,))])

for i, k in enumerate(key_freqs_plus):
    print(f'Cluster {i}: freq k={k}, {(neuron_freqs==k).sum()} neurons')
# %%
fourier_norms_in_each_cluster = []
for freq in key_freqs:
    fourier_norms_in_each_cluster.append(
        einops.reduce(
            neuron_acts_centered_fourier.pow(2)[..., neuron_freqs==freq], 
            'batch_y batch_x neuron -> batch_y batch_x', 
            'mean'
        )
    )

imshow_fourier(
    t.stack(fourier_norms_in_each_cluster), 
    title=f'Norm of 2D Fourier components of neuron activations in each cluster',
    facet_col=0,
    facet_labels=[f"Freq={freq}" for freq in key_freqs]
)
# %%
def project_onto_direction(batch_vecs: t.Tensor, v: t.Tensor) -> t.Tensor:
    '''
    Returns the component of each vector in `batch_vecs` in the direction of `v`.

    batch_vecs.shape = (n, ...)
    v.shape = (n,)
    '''

    # Get tensor of components of each vector in v-direction

    # didn't actually need to normalise lol but it's ok

    v_normalised = t.nn.functional.normalize(v.float(), dim=0)
    dot_prod = einops.einsum(v_normalised, batch_vecs, 'n, n ... ->...')

    projected = einops.einsum(dot_prod, v_normalised, '..., n -> n ...')
    return projected

    # Use these components as coefficients of v in our projections

#%%
# tests.test_project_onto_direction(project_onto_direction)
# %%

def project_onto_frequency(batch_vecs: t.Tensor, freq: int) -> t.Tensor:
    '''
    Returns the projection of each vector in `batch_vecs` onto the
    2D Fourier basis directions corresponding to frequency `freq`.

    batch_vecs.shape = (p**2, ...)
    '''
    assert batch_vecs.shape[0] == p**2

    const = fourier_2d_basis_term(0,0).reshape((p**2,))
    cosx = fourier_2d_basis_term(2*freq,0).reshape((p**2,))
    cosy = fourier_2d_basis_term(0,2*freq).reshape((p**2,))
    sinx = fourier_2d_basis_term(2*freq-1,0).reshape((p**2,))
    siny = fourier_2d_basis_term(0,2*freq-1).reshape((p**2,))
    cosxcosy = fourier_2d_basis_term(2*freq,2*freq).reshape((p**2,))
    cosxsiny = fourier_2d_basis_term(2*freq,2*freq-1).reshape((p**2,))
    sinxsiny = fourier_2d_basis_term(2*freq-1,2*freq-1).reshape((p**2,))
    sinxcosy = fourier_2d_basis_term(2*freq-1,2*freq).reshape((p**2,))
    
    fourier_basis = [const, cosx, cosy, sinx, siny, cosxcosy, cosxsiny, sinxsiny, sinxcosy]

    # projection = 0
    # for basis in fourier_basis:
    #     projection += project_onto_direction(batch_vecs, basis)

    projection = sum([project_onto_direction(batch_vecs, basis) for basis in fourier_basis])
    # directions are vectors of length p**2

    return projection

    # return sum([
    #     project_onto_direction(
    #         batch_vecs,
    #         fourier_2d_basis_term(i, j).flatten(),
    #     )
    #     for i in [0, 2*freq-1, 2*freq] for j in [0, 2*freq-1, 2*freq]
    # ])
#%%
# tests.test_project_onto_frequency(project_onto_frequency)
# %%

logits_in_freqs = []

for freq in key_freqs:

    # Get all neuron activations corresponding to this frequency
    filtered_neuron_acts = neuron_acts_post[:, neuron_freqs==freq]

    # Project onto const/linear/quadratic terms in 2D Fourier basis
    filtered_neuron_acts_in_freq = project_onto_frequency(filtered_neuron_acts, freq)

    # Calcluate new logits, from these filtered neuron activations
    logits_in_freq = filtered_neuron_acts_in_freq @ W_logit[neuron_freqs==freq]

    logits_in_freqs.append(logits_in_freq)

# We add on neurons in the always firing cluster, unfiltered
logits_always_firing = neuron_acts_post[:, neuron_freqs==-1] @ W_logit[neuron_freqs==-1]
logits_in_freqs.append(logits_always_firing)

# Print new losses
print('Loss with neuron activations ONLY in key freq (inclusing always firing cluster)\n{:.6e}\n'.format( 
    test_logits(
        sum(logits_in_freqs), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Loss with neuron activations ONLY in key freq (exclusing always firing cluster)\n{:.6e}\n'.format( 
    test_logits(
        sum(logits_in_freqs[:-1]), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Original loss\n{:.6e}'.format(original_loss))

# %%

print('Loss with neuron activations excluding none:     {:.9f}'.format(original_loss.item()))
for c, freq in enumerate(key_freqs_plus):
    print('Loss with neuron activations excluding freq={}:  {:.9f}'.format(
        freq, 
        test_logits(
            sum(logits_in_freqs) - logits_in_freqs[c], 
            bias_correction=True, 
            original_logits=original_logits
        )
    ))
# %%
imshow_fourier(
    einops.reduce(neuron_acts_centered_fourier.pow(2), 'y x neuron -> y x', 'mean'), 
    title='Norm of Fourier Components of Neuron Acts'
)

# Rearrange logits, so the first two dims represent (x, y) in modular arithmetic equation
original_logits_sq = einops.rearrange(original_logits, "(x y) z -> x y z", x=p)
original_logits_fourier = fft2d(original_logits_sq)

imshow_fourier(
    einops.reduce(original_logits_fourier.pow(2), 'y x z -> y x', 'mean'), 
    title='Norm of Fourier Components of Logits'
)
# %%

def get_trig_sum_directions(k: int) -> Tuple[Float[Tensor, "p p"], Float[Tensor, "p p"]]:
    '''
    Given frequency k, returns the normalized vectors in the 2D Fourier basis 
    representing the directions:

        cos(ω_k * (x + y))
        sin(ω_k * (x + y))

    respectively.
    '''
    # cos(x + y) = cos(x)cos(y) - sin(x)sin(y)
    # sin(x + y) = cos(x)sin(y) + sin(x)cos(y)

    cosxcosy = fourier_2d_basis_term(2*k, 2*k)
    sinxsiny = fourier_2d_basis_term(2*k-1,2*k-1)
    cosxsiny = fourier_2d_basis_term(2*k,2*k-1)
    sinxcosy = fourier_2d_basis_term(2*k-1,2*k)

    cosxy = (cosxcosy - sinxsiny) / np.sqrt(2)
    sinxy = (cosxsiny + sinxcosy) / np.sqrt(2)

    return cosxy, sinxy
#%%

# tests.test_get_trig_sum_directions(get_trig_sum_directions)
# %%
trig_logits = []

for k in key_freqs:

    cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(k)

    cos_xplusy_projection = project_onto_direction(
        original_logits,
        cos_xplusy_direction.flatten()
    )

    sin_xplusy_projection = project_onto_direction(
        original_logits,
        sin_xplusy_direction.flatten()
    )

    trig_logits.extend([cos_xplusy_projection, sin_xplusy_projection])

trig_logits = sum(trig_logits)

print(f'Loss with just x+y components: {test_logits(trig_logits, True, original_logits):.4e}')
print(f"Original Loss: {original_loss:.4e}")
# %%

US = W_logit @ fourier_basis.T

imshow_div(
    US,
    x=fourier_basis_names,
    yaxis='Neuron index',
    title='W_logit in the Fourier Basis',
    height=800,
    width=600
)
# %%
US_sorted = t.concatenate([
    US[neuron_freqs==freq] for freq in key_freqs_plus
])
hline_positions = np.cumsum([(neuron_freqs == freq).sum().item() for freq in key_freqs]).tolist() + [cfg.d_mlp]

imshow_div(
    US_sorted,
    x=fourier_basis_names, 
    yaxis='Neuron',
    title='W_logit in the Fourier Basis (rearranged by neuron cluster)',
    hline_positions = hline_positions,
    hline_labels = [f"Cluster: {freq=}" for freq in key_freqs.tolist()] + ["No freq"],
    height=800,
    width=600
)
# %%

cos_components = []
sin_components = []

for k in key_freqs:
    σu_sin = US[:, 2*k]
    σu_cos = US[:, 2*k-1]

    logits_in_cos_dir = neuron_acts_post_sq @ σu_cos
    logits_in_sin_dir = neuron_acts_post_sq @ σu_sin

    cos_components.append(fft2d(logits_in_cos_dir))
    sin_components.append(fft2d(logits_in_sin_dir))

for title, components in zip(['Cosine', 'Sine'], [cos_components, sin_components]):
    imshow_fourier(
        t.stack(components),
        title=f'{title} components of neuron activations in Fourier basis',
        animation_frame=0,
        animation_name="Frequency",
        animation_labels=key_freqs.tolist()
    )
# %%


epochs = full_run_data['epochs']

# Define a dictionary to store our metrics in
metric_cache = {}


def get_metrics(model: HookedTransformer, metric_cache, metric_fn, name, reset=False):
    '''
    Define a metric (by metric_fn) and add it to the cache, with the name `name`.

    If `reset` is True, then the metric will be recomputed, even if it is already in the cache.
    '''
    if reset or (name not in metric_cache) or (len(metric_cache[name])==0):
        metric_cache[name]=[]
        for c, sd in enumerate(tqdm((full_run_data['state_dicts']))):
            model = load_in_state_dict(model, sd)
            out = metric_fn(model)
            if type(out)==t.Tensor:
                out = utils.to_numpy(out)
            metric_cache[name].append(out)
        model = load_in_state_dict(model, full_run_data['state_dicts'][400])
        try:
            metric_cache[name] = t.tensor(metric_cache[name])
        except:
            metric_cache[name] = t.tensor(np.array(metric_cache[name]))


plot_metric = partial(lines, x=epochs, xaxis='Epoch', log_y=True)


def test_loss(model):
    logits = model(all_data)[:, -1, :-1]
    return test_logits(logits, False, mode='test')

get_metrics(model, metric_cache, test_loss, 'test_loss')


def train_loss(model):
    logits = model(all_data)[:, -1, :-1]
    return test_logits(logits, False, mode='train')

get_metrics(model, metric_cache, train_loss, 'train_loss')
# %%

def excl_loss(model: HookedTransformer, key_freqs: list) -> list:
    '''
    Returns the excluded loss (i.e. subtracting the components of logits corresponding to 
    cos(w_k(x+y)) and sin(w_k(x+y)), for each frequency k in key_freqs.
    '''
    excl_loss_list = []
    logits = model(all_data)[:, -1, :-1]
    for k in key_freqs:
        # returns vectors in 2D fourier space corresponding to cos(w_k(x+y)) and sin(w_k(x+y))
        cosk, sink = get_trig_sum_directions(k)
        # project logits onto these vectors, and subtract this from logits?
        cosk_projection = project_onto_direction(logits, cosk.flatten())
        sink_projection = project_onto_direction(logits, sink.flatten())
        logits_excl = logits - cosk_projection - sink_projection
        loss = test_logits(logits_excl, bias_correction=False, mode='train').item()
        excl_loss_list.append(loss)
    return excl_loss_list     
#%%

tests.test_excl_loss(excl_loss, model, key_freqs)
# %%
excl_loss = partial(excl_loss, key_freqs=key_freqs)
get_metrics(model, metric_cache, excl_loss, 'excl_loss')

lines(
    t.concat([
        metric_cache['excl_loss'].T, 
        metric_cache['train_loss'][None, :],  
        metric_cache['test_loss'][None, :]
    ], axis=0), 
    labels=[f'excl {freq}' for freq in key_freqs]+['train', 'test'], 
    title='Excluded Loss for each trig component',
    log_y=True,
    x=full_run_data['epochs'],
    xaxis='Epoch',
    yaxis='Loss'
)
# %%
def fourier_embed(model: HookedTransformer):
    '''
    Returns norm of Fourier transform of the model's embedding matrix.
    '''
    # embed = model.W_E[:-1]
    # transformed_embed = fourier_basis.T @ embed
    # #return transformed_embed.norm(dim=-1)
    # return einops.reduce(transformed_embed.pow(2), 'p e -> p', 'sum')
    W_E_fourier = fourier_basis.T @ model.W_E[:-1]
    return einops.reduce(W_E_fourier.pow(2), 'vocab d_model -> vocab', 'sum')

tests.test_fourier_embed(fourier_embed, model)
# %%

# Plot every 200 epochs so it's not overwhelming
get_metrics(model, metric_cache, fourier_embed, 'fourier_embed')

animate_lines(
    metric_cache['fourier_embed'][::2],
    snapshot_index = epochs[::2],
    snapshot='Epoch',
    hover=fourier_basis_names,
    animation_group='x',
    title='Norm of Fourier Components in the Embedding Over Training',
)
# %%
def embed_SVD(model: HookedTransformer) -> t.Tensor:
    '''
    Returns vector S, where W_E = U @ diag(S) @ V.T in singular value decomp.
    '''
    _, S, _ = t.svd(model.W_E[:,:-1])
    return S


tests.test_embed_SVD(embed_SVD, model)
# %%
get_metrics(model, metric_cache, embed_SVD, 'embed_SVD')

animate_lines(
    metric_cache['embed_SVD'],
    snapshot_index = epochs,
    snapshot='Epoch',
    title='Singular Values of the Embedding During Training',
    xaxis='Singular Number',
    yaxis='Singular Value',
)
# %%

def tensor_trig_ratio(model: HookedTransformer, mode: str):
    '''
    Returns the fraction of variance of the (centered) activations which
    is explained by the Fourier directions corresponding to cos(ω(x+y))
    and sin(ω(x+y)) for all the key frequencies.
    '''
    logits, cache = model.run_with_cache(all_data)
    logits = logits[:, -1, :-1]

    if mode == "neuron_pre":
        tensor = cache['pre', 0][:, -1]
    elif mode == "neuron_post":
        tensor = cache['post', 0][:, -1]
    elif mode == "logit":
        tensor = logits
    else:
        raise ValueError(f"{mode} is not a valid mode")

    # Python 3.10 has match/case statements (-:
    # match mode:
    #     case 'neuron_pre': tensor = cache['pre', 0][:, -1]
    #     case 'neuron_post': tensor = cache['post', 0][:, -1]
    #     case 'logit': tensor = logits
    #     case _: raise ValueError(f"{mode} is not a valid mode")

    tensor_centered = tensor - einops.reduce(tensor, 'xy index -> 1 index', 'mean')
    tensor_var = einops.reduce(tensor_centered.pow(2), 'xy index -> index', 'sum')
    tensor_trig_vars = []

    for freq in key_freqs:
        cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(freq)
        cos_xplusy_projection_var = project_onto_direction(
            tensor_centered, cos_xplusy_direction.flatten()
        ).pow(2).sum(0)
        sin_xplusy_projection_var = project_onto_direction(
            tensor_centered, sin_xplusy_direction.flatten()
        ).pow(2).sum(0)

        tensor_trig_vars.extend([cos_xplusy_projection_var, sin_xplusy_projection_var])

    return utils.to_numpy(sum(tensor_trig_vars)/tensor_var)



for mode in ['neuron_pre', 'neuron_post', 'logit']:
    get_metrics(
        model, 
        metric_cache, 
        partial(tensor_trig_ratio, mode=mode), 
        f"{mode}_trig_ratio", 
        reset=True
    )

lines_list = []
line_labels = []
for mode in ['neuron_pre', 'neuron_post', 'logit']:
    tensor = metric_cache[f"{mode}_trig_ratio"]
    lines_list.append(einops.reduce(tensor, 'epoch index -> epoch', 'mean'))
    line_labels.append(f"{mode}_trig_frac")

plot_metric(
    lines_list, 
    labels=line_labels, 
    log_y=False,
    yaxis='Ratio',
    title='Fraction of logits and neurons explained by trig terms',
)
# %%
def get_frac_explained(model: HookedTransformer):
    _, cache = model.run_with_cache(all_data, return_type=None)

    returns = []

    for neuron_type in ['pre', 'post']:
        neuron_acts = cache[neuron_type, 0][:, -1].clone().detach()
        neuron_acts_centered = neuron_acts - neuron_acts.mean(0)
        neuron_acts_fourier = fft2d(
            einops.rearrange(neuron_acts_centered, "(x y) neuron -> x y neuron", x=p)
        )

        # Calculate the sum of squares over all inputs, for each neuron
        square_of_all_terms = einops.reduce(
            neuron_acts_fourier.pow(2), "x y neuron -> neuron", "sum"
        )

        frac_explained = t.zeros(d_mlp).to(device)
        frac_explained_quadratic_terms = t.zeros(d_mlp).to(device)

        for freq in key_freqs_plus:
            # Get Fourier activations for neurons in this frequency cluster
            # We arrange by frequency (i.e. each freq has a 3x3 grid with const, linear & quadratic terms)
            acts_fourier = arrange_by_2d_freqs(neuron_acts_fourier[..., neuron_freqs==freq])

            # Calculate the sum of squares over all inputs, after filtering for just this frequency
            # Also calculate the sum of squares for just the quadratic terms in this frequency
            if freq==-1:
                squares_for_this_freq = squares_for_this_freq_quadratic_terms = einops.reduce(
                    acts_fourier[:, 1:, 1:].pow(2), "freq x y neuron -> neuron", "sum"
                )
            else:
                squares_for_this_freq = einops.reduce(
                    acts_fourier[freq-1].pow(2), "x y neuron -> neuron", "sum"
                )
                squares_for_this_freq_quadratic_terms = einops.reduce(
                    acts_fourier[freq-1, 1:, 1:].pow(2), "x y neuron -> neuron", "sum"
                )

            frac_explained[neuron_freqs==freq] = squares_for_this_freq / square_of_all_terms[neuron_freqs==freq]
            frac_explained_quadratic_terms[neuron_freqs==freq] = squares_for_this_freq_quadratic_terms / square_of_all_terms[neuron_freqs==freq]

        returns.extend([frac_explained, frac_explained_quadratic_terms])

    frac_active = (neuron_acts > 0).float().mean(0)

    return t.nan_to_num(t.stack(returns + [neuron_freqs, frac_active], axis=0))



get_metrics(model, metric_cache, get_frac_explained, 'get_frac_explained')

frac_explained_pre = metric_cache['get_frac_explained'][:, 0]
frac_explained_quadratic_pre = metric_cache['get_frac_explained'][:, 1]
frac_explained_post = metric_cache['get_frac_explained'][:, 2]
frac_explained_quadratic_post = metric_cache['get_frac_explained'][:, 3]
neuron_freqs_ = metric_cache['get_frac_explained'][:, 4]
frac_active = metric_cache['get_frac_explained'][:, 5]

animate_scatter(
    t.stack([frac_explained_quadratic_pre, frac_explained_quadratic_post], dim=1)[:200:5],
    color=neuron_freqs_[:200:5], 
    color_name='freq',
    snapshot='epoch',
    snapshot_index=epochs[:200:5],
    xaxis='Quad ratio pre',
    yaxis='Quad ratio post',
    color_continuous_scale='viridis',
    title='Fraction of variance explained by quadratic terms (up to epoch 20K)'
)

animate_scatter(
    t.stack([neuron_freqs_, frac_explained_pre, frac_explained_post], dim=1)[:200:5],
    color=frac_active[:200:5],
    color_name='frac_active',
    snapshot='epoch',
    snapshot_index=epochs[:200:5],
    xaxis='Freq',
    yaxis='Frac explained',
    hover=list(range(d_mlp)),
    color_continuous_scale='viridis',
    title='Fraction of variance explained by this frequency (up to epoch 20K)'
)
# %%
