#%%
import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")
#%%
if MAIN:
    display_array_as_img(arr[0])
# %%
arr1 = arr[0]

if MAIN:
    display_array_as_img(arr1)

#%%
arr2 = einops.rearrange(arr, 'number channel height width -> channel height (number width)')

if MAIN:
    display_array_as_img(arr2)
#%%
arr3 = einops.repeat(arr[0], 'channel height width -> channel (2 height) width')

if MAIN:
    display_array_as_img(arr3)
#%%
arr4 = einops.repeat(arr[:2], 'num chan h w -> chan (num h) (2 w)')

if MAIN:
    display_array_as_img(arr4)
#%%
arr5 = einops.repeat(arr[0], 'chan h w -> chan (h 2) w')

if MAIN:
    display_array_as_img(arr5)
#%%
arr6 = einops.rearrange(arr[0], 'chan h w -> h (chan w)')

if MAIN:
    display_array_as_img(arr6)
#%%
arr7 = einops.rearrange(arr, '(row col) chan h w -> chan (row h) (col w)', col=3)

if MAIN:
    display_array_as_img(arr7)
#%%
arr8 = einops.reduce(arr, 'num chan h w -> h (num w)', 'max')

if MAIN:
    display_array_as_img(arr8)
#%%
arr9 = einops.reduce(arr.astype(float), 'num chan h w -> h (num w)', 'mean')

if MAIN:
    display_array_as_img(arr9)
#%%
arr10 = einops.reduce(arr, 'num chan h w -> h w', 'min')

if MAIN:
    display_array_as_img(arr10)

#%% 

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, 'i i -> ')

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, 'n m, m -> n')

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, 'n m, m p -> n p')

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, 'n, n -> ')

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, 'n, m -> n m')

if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)
# %%

if MAIN:
    test_input = t.tensor(
        [[0, 1, 2, 3, 4], 
        [5, 6, 7, 8, 9], 
        [10, 11, 12, 13, 14], 
        [15, 16, 17, 18, 19]], dtype=t.float
    )
    print(test_input.stride())


#%%

import torch as t
from collections import namedtuple


if MAIN:
    TestCase = namedtuple("TestCase", ["output", "size", "stride"])

    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]), 
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 2], [5, 7]]), 
            size=(2, 2),
            stride=(5, 2),
        ),

        TestCase(
            output=t.tensor([0, 1, 2, 3, 4]),
            size=(5, ),
            stride=(1,),
        ),

        TestCase(
            output=t.tensor([0, 5, 10, 15]),
            size=(4,),
            stride=(5,),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [5, 6, 7]
            ]), 
            size=(2, 3),
            stride=(5, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2, 3),
            stride=(10, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2, 3),
            stride=(11, 0),
        ),

        TestCase(
            output=t.tensor([0, 6, 12, 18]), 
            size=(4, ),
            stride=(6,),
        ),
    ]

    for (i, test_case) in enumerate(test_cases):
        if (test_case.size is None) or (test_case.stride is None):
            print(f"Test {i} failed: attempt missing.")
        else:
            actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
            if (test_case.output != actual).any():
                print(f"Test {i} failed:")
                print(f"Expected: {test_case.output}")
                print(f"Actual: {actual}\n")
            else:
                print(f"Test {i} passed!\n")

# %%
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    row, col = mat.shape
    stride = vec.stride(dim=0)
    vec_strided = vec.as_strided((row, col), (0, stride))
    return (mat * vec_strided).sum(dim=1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)
# %%
def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    row, shared_dim = matA.shape
    col = matB.shape[1]
    a_strided = matA.as_strided((row, shared_dim, col), (*matA.stride(), 0))
    b_strided = matB.as_strided((row, shared_dim, col), (0, *matB.stride()))
    return (a_strided * b_strided).sum(dim=1)


if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)
#%%
def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    x_len = x.shape[0]
    w_len = weights.shape[0]
    stide_size = x.stride(dim=0)
    strided = x.as_strided((x_len-w_len+1, w_len), (stide_size,stide_size))
    return (strided * weights).sum(dim=-1)


if MAIN:
    tests.test_conv1d_minimal_simple(conv1d_minimal_simple)
# %%
def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    x_len = x.shape[-1]
    w_len = weights.shape[-1]
    out_len = x_len-w_len+1
    strided = x.as_strided(
        (*x.shape[:-1], out_len, w_len),
        (*x.stride(), x.stride(-1)))
    
    return einops.einsum(strided, weights,
                         'batch in_ch out_w k_w, out_ch in_ch k_w -> batch out_ch out_w')


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)
# %%

def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    x_h, x_w = x.shape[-2:]
    w_h, w_w = weights.shape[-2:]
    out_h, out_w = x_h - w_h + 1, x_w - w_w + 1
    strided = x.as_strided(
        (*x.shape[:-2], out_h, out_w, w_h, w_w),
        (*x.stride(), *x.stride()[-2:]))
    
    return einops.einsum(strided, weights,
                         'batch in_ch out_h out_w w_h w_w, out_ch in_ch w_h w_w ->' + 
                         ' batch out_ch out_h out_w')

if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)
# %%

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    out_width = x.shape[-1] + left + right
    out = x.new_full((*x.shape[:-1], out_width), pad_value)
    out[..., left:out_width - right] = x
    return out

if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)
# %%

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    out_height, out_width = x.shape[-2] + top + bottom , x.shape[-1] + left + right
    out = x.new_full((*x.shape[:-2], out_height, out_width), pad_value)
    out[..., top: out_height - bottom, left:out_width - right] = x
    return out



if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)
# %%

def conv1d(
    x: Float[Tensor, "b ic w"], 
    weights: Float[Tensor, "oc ic kw"], 
    stride: int = 1, 
    padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    out_len = (x.shape[-1] + 2*padding - weights.shape[-1])//stride + 1
    x_pad = pad1d(x, padding, padding, 0.0)
    strided = x_pad.as_strided(
        (*x_pad.shape[:-1], out_len, weights.shape[-1]),
        (*x_pad.stride()[:-1], stride*x_pad.stride(-1), x_pad.stride(-1)))
    
    return einops.einsum(strided, weights,
                         'batch in_ch out_w k_w, out_ch in_ch k_w -> batch out_ch out_w')


if MAIN:
    tests.test_conv1d(conv1d)
# %%
IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:


if MAIN:
    for v in [(1, 2), 2, (1, 2, 3)]:
        try:
            print(f"{v!r:9} -> {force_pair(v)!r}")
        except ValueError:
            print(f"{v!r:9} -> ValueError")
# %%
def conv2d(
    x: Float[Tensor, "b ic h w"], 
    weights: Float[Tensor, "oc ic kh kw"], 
    stride: IntOrPair = 1, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    stride = force_pair(stride)
    padding = force_pair(padding)

    out_h = (x.shape[-2] + 2*padding[0] - weights.shape[-2])//stride[0] + 1
    out_w = (x.shape[-1] + 2*padding[1] - weights.shape[-1])//stride[1] + 1
    x_pad = pad2d(x, padding[1], padding[1], padding[0], padding[0], 0.0)
    strided = x_pad.as_strided(
        (*x_pad.shape[:-2], out_h, out_w, *weights.shape[-2:]),
        (*x_pad.stride()[:-2], stride[0]*x_pad.stride(-2),
         stride[1]*x_pad.stride(-1), *x_pad.stride()[-2:]))
    
    return einops.einsum(strided, weights,
                         'batch in_ch out_h out_w k_h k_w, out_ch in_ch k_h k_w'+
                         ' -> batch out_ch out_h out_w') 


if MAIN:
    tests.test_conv2d(conv2d)
# %%
def maxpool2d(
    x: Float[Tensor, "b ic h w"], 
    kernel_size: IntOrPair, 
    stride: Optional[IntOrPair] = None, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
    '''
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    kernel_size = force_pair(kernel_size)
    if stride == None: stride = kernel_size
    else: stride = force_pair(stride)
    padding = force_pair(padding)
    
    out_h = (x.shape[-2] + 2*padding[0] - kernel_size[0])//stride[0] + 1
    out_w = (x.shape[-1] + 2*padding[1] - kernel_size[1])//stride[1] + 1
    x_pad = pad2d(x, padding[1], padding[1], padding[0], padding[0], float('-inf'))
    strided = x_pad.as_strided(
        (*x_pad.shape[:-2], out_h, out_w, *kernel_size),
        (*x_pad.stride()[:-2], stride[0]*x_pad.stride(-2),
         stride[1]*x_pad.stride(-1), *x_pad.stride()[-2:]))
    
    return einops.reduce(strided,
                         'batch chan out_h out_w k_h k_w -> batch chan out_h out_w',
                         'max') 


if MAIN:
    tests.test_maxpool2d(maxpool2d)
# %%
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = force_pair(kernel_size)
        if stride is None: self.stride = kernel_size
        else: self.stride = force_pair(stride)
        self.padding = force_pair(padding)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        param_list = [f'{k} {getattr(self, k)}' for k in ['kernel_size', 'stride', 'padding']]
        return f"Layer with parameters: {' '.join(param_list)}"


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")
# %%

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.clamp(min=0)


if MAIN:
    tests.test_relu(ReLU)
# %%

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        shape = (*input.shape[:self.start_dim], -1)
        if self.end_dim != -1:
            shape = (*shape, *input.shape[self.end_dim+1:]) 
        return input.view(shape)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])



if MAIN:
    tests.test_flatten(Flatten)
# %%

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        init_scale = 1/float(in_features)**0.5
        self.weight = nn.Parameter(t.empty(out_features, in_features,
                                          dtype=t.float32).uniform_(-init_scale, 
                                                                    init_scale))
        if bias:
            self.bias = nn.Parameter(t.zeros(out_features, dtype=t.float32))
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        # assert x.shape[0] == self.weight.shape[1]
        out = einops.einsum(self.weight, x, 'out_dim in_dim, ... in_dim -> ... out_dim')
        if self.bias is None:
            return out
        return out + self.bias 

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["weight", "bias"]])


if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)
# %%
