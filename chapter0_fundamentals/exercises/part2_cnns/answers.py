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

# %%
if MAIN:
    display_array_as_img(arr[0])

# Exercise 2
# %%
if MAIN:
    arr2 = einops.rearrange(arr, "b c h w -> c h (b w)")
    display_array_as_img(arr2)

# Exercise 3
# %%
if MAIN:
    arr3 = einops.repeat(arr[0], "c h w -> c (n h) w", n=2)
    display_array_as_img(arr3)

# Exercise 4
# %%
if MAIN:
    arr4 = einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")
    display_array_as_img(arr4)

# Exercise 5
# %%
if MAIN:
    arr5 = einops.repeat(arr[0], "c h w -> c (h 2) w")
    display_array_as_img(arr5)

# Exercise 6
# %%
if MAIN:
    arr6 = einops.rearrange(arr[0], "c h w -> h (c w)")
    display_array_as_img(arr6)

# Exercise 7
# %%
if MAIN:
    arr7 = einops.rearrange(arr, "(bh bw) c h w -> c (bh h) (bw w)", bh=2, bw=3)
    display_array_as_img(arr7)

# Exercise 8
# %%
if MAIN:
    arr8 = einops.reduce(arr, "b c h w -> h (b w)", "max")
    display_array_as_img(arr8)

# Exercise 9
# %%
if MAIN:
    arr9 = einops.reduce(arr, "b c h w -> h (b w)", "min")
    display_array_as_img(arr9)

# Exercise 10
# %%
if MAIN:
    arr10 = einops.reduce(arr, "b c h w -> h w", "min")
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
    return einops.einsum(mat, vec, "i j, j -> i")

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, "i j, j k -> i k")

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, "i,i ->")

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, "i, j -> i j")


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
            stride=(1,)
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
            stride=(5, 1)
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2, 3),
            stride=(10, 1)
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2, 3),
            stride=(11, 0)
        ),

        TestCase(
            output=t.tensor([0, 6, 12, 18]), 
            size=(4,),
            stride=(6,)
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
    stride_row, stride_column = mat.stride()
    size = (min(mat.shape),)
    stride = ((stride_row + stride_column),)
    diagonal = mat.as_strided(size=size, stride=stride)
    return diagonal.sum()


if MAIN:
    tests.test_trace(as_strided_trace)

# %%
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    size = mat.shape
    stride = (0, vec.stride(0))
    vec_repeat = vec.as_strided(size=size, stride=stride)
    return (mat * vec_repeat).sum(dim=1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)

# %%

def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    left, right, inner = matA.shape[0], matB.shape[1], matA.shape[1]

    A_strided = matA.as_strided(size=(left, right, inner), stride=(matA.stride(0), 0, matA.stride(1)))
    B_strided = matB.as_strided(size=(left, right, inner), stride=(0, matB.stride(1), matB.stride(0)))

    return (A_strided * B_strided).sum(dim=-1)


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
    width = x.shape[0]
    kernel_width = weights.shape[0]
    output_width = width - kernel_width + 1

    x_strided = x.as_strided(
        size=(output_width, kernel_width),
        stride=(x.stride(0), x.stride(0)),
    )

    return einops.einsum(x_strided, weights, "ow kw, kw -> ow")

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
    batch, input_channels, width = x.shape
    output_channels, input_channels, kernel_width = weights.shape
    output_width = width - kernel_width + 1

    x_strided = x.as_strided(
        size=(batch, input_channels, output_width, kernel_width),
        stride=(x.stride(0), x.stride(1), x.stride(2), x.stride(2)),
    )

    return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")

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
    batch, input_channels, height, width = x.shape
    output_channels, input_channels, kernel_height, kernel_width = weights.shape
    output_width = width - kernel_width + 1
    output_height = height - kernel_height + 1

    s_batch, s_input_channels, s_height, s_width = x.stride()

    x_strided = x.as_strided(
        size=(batch, input_channels, output_height, kernel_height, output_width, kernel_width),
        stride=(s_batch, s_input_channels, s_height, s_height, s_width, s_width),
    )

    return einops.einsum(x_strided, weights, "b ic oh kh ow kw, oc ic kh kw -> b oc oh ow")


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)

# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    batch, in_channels, width = x.shape
    x_padded = x.new_full((batch, in_channels, left + right + width), pad_value)
    x_padded[..., left:left+width] = x
    return x_padded

if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)

# %%
def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    batch, in_channels, height, width = x.shape
    x_padded = x.new_full(
        size=(batch, in_channels, top + bottom + height, left + right + width),
        fill_value=pad_value,
    )
    x_padded[..., top:top+height,left:left+width] = x
    return x_padded

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
    batch, input_channels, width = x.shape
    output_channels, input_channels, kernel_width = weights.shape
    output_width = (width + 2 * padding - kernel_width) // stride + 1

    x_padded = pad1d(x, padding, padding, 0)
    s_batch, s_input_channels, s_width = x_padded.stride()
    x_strided = x_padded.as_strided(
        size=(batch, input_channels, output_width, kernel_width),
        stride=(s_batch, s_input_channels, s_width * stride, s_width),
    )

    return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")

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
    padding_height, padding_width = force_pair(padding)
    stride_down, stride_right = force_pair(stride)

    batch, input_channels, height, width = x.shape
    output_channels, input_channels, kernel_height, kernel_width = weights.shape
    output_height = (height + 2 * padding_height - kernel_height) // stride_down + 1
    output_width = (width + 2 * padding_width - kernel_width) // stride_right + 1

    x_padded = pad2d(x, padding_width, padding_width, padding_height, padding_height, 0)
    s_batch, s_input_channels, s_height, s_width = x_padded.stride()
    x_strided = x_padded.as_strided(
        size=(batch, input_channels, output_height, kernel_height ,output_width, kernel_width),
        stride=(s_batch, s_input_channels, s_height * stride_down, s_height, s_width * stride_right, s_width),
    )

    return einops.einsum(x_strided, weights, "b ic oh kh ow kw, oc ic kh kw -> b oc oh ow")


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
    stride = kernel_size if not stride else stride
    kernel_height, kernel_width = force_pair(kernel_size)
    stride_down, stride_right = force_pair(stride)
    padding_height, padding_width = force_pair(padding)

    batch, input_channels, height, width = x.shape
    output_height = (height + 2 * padding_height - kernel_height) // stride_down + 1
    output_width = (width + 2 * padding_width - kernel_width) // stride_right + 1

    x_padded = pad2d(x, padding_width, padding_width, padding_height, padding_height, -t.inf)
    s_batch, s_input_channels, s_height, s_width = x_padded.stride()
    x_strided = x_padded.as_strided(
        size=(batch, input_channels, output_height, kernel_height ,output_width, kernel_width),
        stride=(s_batch, s_input_channels, s_height * stride_down, s_height, s_width * stride_right, s_width),
    )

    return einops.reduce(x_strided, "b ic oh kh ow kw -> b ic oh ow", "max")


if MAIN:
    tests.test_maxpool2d(maxpool2d)