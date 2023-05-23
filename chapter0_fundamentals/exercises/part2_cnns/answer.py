# %%

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

if MAIN:
    display_array_as_img(arr[0])
# %%
# Your code here - define arr1
arr1 = einops.rearrange(arr, "b c h w -> c h (b w)")

if MAIN:
    display_array_as_img(arr1)
# %%
arr3 = einops.repeat(arr[0], 'c h w -> c (d h) w', d=2)
if MAIN:
    display_array_as_img(arr3)
# %%
arr4 = einops.rearrange(arr[:2], "b c h w -> c (b h) w")
arr4 = einops.repeat(arr4, 'c h w -> c h (d w)', d=2)
if MAIN:
    display_array_as_img(arr4)
# %%
arr5 = einops.repeat(arr[0], 'c h w -> c (h d) w', d=2)
if MAIN:
    display_array_as_img(arr5)
# %%
arr6 = einops.repeat(arr[0], 'c h w -> h (c w)')
if MAIN:
    display_array_as_img(arr6)
# %%
arr7 = einops.repeat(arr, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2)
if MAIN:
    display_array_as_img(arr7)
# %%
arr8 = einops.reduce(arr, 'b c h w -> h (b w)', 'max')

if MAIN:
    display_array_as_img(arr8)
# %%
arr9 = einops.reduce(arr.astype('float'), 'b c h w -> h (b w)', 'mean').astype('int')

if MAIN:
    display_array_as_img(arr9)

# %%
arr10 = einops.reduce(arr, 'b c h w -> h (w)', 'min')

if MAIN:
    display_array_as_img(arr10)

# %%
def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, "i i ->")

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, 'i j, j -> i')

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, 'i j, j k -> i k')

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, 'i,i->')

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, 'i,j->i j')

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
# %%
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
            size=(5,),
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
            size=(4,),
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
def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    nrows, ncols = mat.shape
    return mat.as_strided(size=(min(nrows, ncols),), stride=(ncols+1,)).sum()
    return


if MAIN:
    tests.test_trace(as_strided_trace)
# %%
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    mat_rows, mat_cols = mat.shape
    vec_dim = mat_cols
    vec_stride_d1 = vec.stride()[0]
    print(vec_stride_d1)
    # print(vec)
    vec_expanded = vec.as_strided(size=(mat_rows, mat_cols), stride=(0,vec_stride_d1))
    # print(vec_expanded)
    something = mat * vec_expanded
    return something.sum(1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)
# %%
def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    i, j = matA.shape
    _, k = matB.shape
    Ai, Aj = matA.stride()
    Bj, Bk = matB.stride()
    # print(matA)
    # print(matB)
    # print(matA.shape)
    # print(matA)
    matA_ext = matA.as_strided(size=(i, j, k), stride=(Ai, Aj, 0))
    matB_ext = matB.as_strided(size=(i, j, k), stride=(0, Bj, Bk))
    out = matA_ext * matB_ext
    return out.sum(dim=1)



if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)
# %%
def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    kw = weights.shape[0]
    w = x.shape[0]
    ow = w - kw + 1

    # print(kw)
    x_ext = x.as_strided(size=(ow, kw), stride=(1, 1))
    w_ext = weights.as_strided(size=(ow, kw), stride=(0, 1))
    # print(x)
    # print(x_ext)
    # print(weights)
    # print(w_ext)

    out = x_ext * w_ext
    return out.sum(dim=-1)

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
    b, ic, w = x.shape
    oc, ic_2, kw = weights.shape
    ow = w - kw + 1

    assert ic == ic_2, "input and kernel shapes not compatible"

    x_ext = x.as_strided(size=(b, ow, ic, kw), stride=(ic*w, 1, w, 1))
    out = einops.einsum(x_ext, weights, 'b ow ic kw, oc ic kw -> b oc ow')
    return out
    # print(weights_ext)


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)
# %%
kernel = t.tensor([[
    [1, 2],
    [3, 4]
], [
    [5, 6],
    [7, 8]
]])
input = t.tensor([[
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]])
conv1d_minimal(input, kernel)

# %%

IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    # SOLUTION

    B, C, H, W = x.shape
    output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
    output[..., top : top + H, left : left + W] = x
    return output


def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)


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
    # SOLUTION

    # Set actual values for stride and padding, using force_pair function
    if stride is None:
        stride = kernel_size
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    kh, kw = force_pair(kernel_size)

    # Get padded version of x
    x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=-t.inf)

    # Calculate output height and width for x
    b, ic, h, w = x_padded.shape
    ow = 1 + (w - kw) // stride_w
    oh = 1 + (h - kh) // stride_h

    # Get strided x
    s_b, s_c, s_h, s_w = x_padded.stride()

    x_new_shape = (b, ic, oh, ow, kh, kw)
    x_new_stride = (s_b, s_c, s_h * stride_h, s_w * stride_w, s_h, s_w)
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)

    # Argmax over dimensions of the maxpool kernel
    # (note these are the same dims that we multiply over in 2D convolutions)
    output = t.amax(x_strided, dim=(-1, -2))
    return output

if MAIN:
    tests.test_maxpool2d(maxpool2d)
# %%

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return f'kernel size: {self.kernel_size}, stride: {self.stride}, padding: {self.padding}'


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")

# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(t.zeros(x.shape), x)


if MAIN:
    tests.test_relu(ReLU)
# %%
from functools import reduce

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        start_dim = self.start_dim if 0 <= self.start_dim else len(input.shape) + self.start_dim
        end_dim = self.end_dim if 0 <= self.end_dim else len(input.shape) + self.end_dim
        new_shape = (
            *input.shape[:start_dim],
            # reduce(lambda a, b: a * b, input.shape[start_dim:end_dim+1], 1),
            -1,
            *input.shape[end_dim+1:],
        )
        return t.reshape(input, new_shape)

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
        self.weight = nn.Parameter((t.rand(out_features, in_features) * (2 / (in_features ** 0.5))) -  (1 / (in_features ** 0.5))) 
        self.bias = nn.Parameter((t.rand(out_features) * (2 / (in_features ** 0.5))) -  (1 / (in_features ** 0.5))) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        if self.bias is not None:
            return t.matmul(x, self.weight.T) + self.bias
        return t.matmul(x, self.weight.T)

    def extra_repr(self) -> str:
        return f'Bias is {self.bias is not None}, in_features: {self.in_features}, out_features: {self.out_features}'

if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)
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
    # SOLUTION

    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)

    x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=0)

    b, ic, h, w = x_padded.shape
    oc, ic2, kh, kw = weights.shape
    assert ic == ic2, "in_channels for x and weights don't match up"
    ow = 1 + (w - kw) // stride_w
    oh = 1 + (h - kh) // stride_h

    s_b, s_ic, s_h, s_w = x_padded.stride()

    # Get strided x (new height/width dims have same stride as original height/width-strides of x, scaled by stride)
    x_new_shape = (b, ic, oh, ow, kh, kw)
    x_new_stride = (s_b, s_ic, s_h * stride_h, s_w * stride_w, s_h, s_w)
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)

    return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = force_pair(stride)
        self.padding = force_pair(padding)

        nparams = self.kernel_size[0] * self.kernel_size[1] * in_channels
        self.weight = nn.Parameter((t.rand(out_channels, in_channels, *self.kernel_size) * (2 / (nparams ** 0.5))) -  (1 / (nparams ** 0.5)))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, weights=self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["in_channels", "out_channels", "kernel_size", "stride", "padding"]])


if MAIN:
    tests.test_conv2d_module(Conv2d)
# %%
def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape
    oh, ow = h - kh + 1, w - kw + 1
    assert ic == ic2, "input and kernel shapes not compatible"

    s0, s1, s2, s3 = x.stride()
    # x_ext = x.as_strided(size=(b, oh, ow, ic, kh, kw), stride=(ic*h*w, w, 1, h*w, w, 1))
    # x_ext = x.as_strided(size=(b, oh, ow, ic, kh, kw), stride=(s0, s2, s3, s1, s2, s3))
    x_ext = x.as_strided(size=(b, ic, oh, ow, kh, kw), stride=(s0, s1, s2, s3, s2, s3))
    # return einops.einsum(x_ext, weights, 'b oh ow ic kh kw, oc ic kh kw -> b oc oh ow')
    return einops.einsum(x_ext, weights, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)
# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(in_features=32*14*14, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.fc(self.flatten(self.relu(self.maxpool(self.conv(x)))))



if MAIN:
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    print(model)

