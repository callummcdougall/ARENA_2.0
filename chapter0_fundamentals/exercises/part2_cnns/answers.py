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
    display_array_as_img(arr[1])

# %%

arr1 = arr[0]

if MAIN:
    display_array_as_img(arr1)

# %%

arr2 = einops.rearrange(arr, 'nums channels y x -> channels y (nums x)')

if MAIN:
    display_array_as_img(arr2)
# %%

arr3 = einops.repeat(arr[0], 'channels y x -> channels (2 y) x')

if MAIN:
    display_array_as_img(arr3)
# %%

arr4 = einops.repeat(arr[:2], 'nums channels y x -> channels (nums y) (2 x)')

if MAIN:
    display_array_as_img(arr4)

# %%

arr5 = einops.repeat(arr[0], 'channels y x -> channels (y 2) x')

if MAIN:
    display_array_as_img(arr5)

# %%

arr6 = einops.rearrange(arr[0], 'channels y x -> y (channels x)')


if MAIN:
    display_array_as_img(arr6)

# %%

arr7 = einops.rearrange(arr, '(a b) channels y x -> channels (a y) (b x)', a=2, b=3)


if MAIN:
    display_array_as_img(arr7)

# %%

arr8 = einops.reduce(arr, 'nums channels y x -> y (nums x)', 'max')
if MAIN:
    display_array_as_img(arr8)

# %%


arr9 = einops.reduce(arr.astype(float), 'nums channels y x -> y (nums x)', 'min')
if MAIN:
    display_array_as_img(arr9)


# %%

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, "i i -> ")

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
    return einops.einsum(vec1, vec2, "i, i ->")

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
    assert mat.size(0) == mat.size(1)

    size = mat.shape[0]
    stride = mat.stride()
    return mat.as_strided(size=(size,), stride=(stride[0] + stride[1],)).sum()


if MAIN:
    tests.test_trace(as_strided_trace)

# %%

def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    i, j = mat.shape
    mat_2 = vec.as_strided((i, j), (0, vec.stride()[0]))
    mul = mat * mat_2
    return mul.sum(-1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)

# %%

def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    i, j, k = (*matA.shape, matB.shape[1])
    size = (i, k, j)
    # Strides represent i, k, js.  
    matA_2 = matA.as_strided(size, (matA.stride()[-2], 0, matA.stride()[-1]))
    matB_2 = matB.as_strided(size, (0, matB.stride()[-1], matB.stride()[-2]))

    mul = matA_2 * matB_2
    return mul.sum(-1)


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
    w, kw = (x.shape[0], weights.shape[0])
    ow = w - kw + 1
    x_part = x.as_strided(size=(ow, kw), stride=(x.stride()[-1], x.stride()[-1]))
    k_part = weights.as_strided(size=(ow, kw), stride=(0, weights.stride()[-1]))
    return t.sum(x_part*k_part, dim=1)


if MAIN:
    tests.test_conv1d_minimal_simple(conv1d_minimal_simple)


# %%

def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
    '''
    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    b, ic, w = x.shape
    oc, ic2, kw = weights.shape
    assert ic == ic2
    ow = w - kw + 1

    x_size = (b, oc, ic, ow, kw)
    x_stride = (x.stride()[-3], 0, x.stride()[-2], x.stride()[-1], x.stride()[-1])
    k_stride = (0, weights.stride()[-3], weights.stride()[-2], 0, weights.stride()[-1])

    x_part = x.as_strided(size=x_size, stride=x_stride)
    k_part = weights.as_strided(size=x_size, stride=k_stride)
    return einops.einsum(x_part*k_part, "b oc ic ow kw -> b oc ow")


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
    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape
    assert ic == ic2
    ow = w - kw + 1
    oh = h - kh + 1

    x_size = (b,
              oc,
              ic,
              oh,
              kh,
              ow,
              kw)
    x_stride = (x.stride()[-3],
                0,
                x.stride()[-4],
                x.stride()[-2],
                x.stride()[-2],
                x.stride()[-1],
                x.stride()[-1])
    k_stride = (0,
                weights.stride()[-3],
                weights.stride()[-4],
                0,
                weights.stride()[-2],
                0,
                weights.stride()[-1])

    x_part = x.as_strided(size=x_size, stride=x_stride)
    k_part = weights.as_strided(size=x_size, stride=k_stride)
    return einops.einsum(x_part*k_part, "b oc ic oh kh ow kw -> b oc oh ow")

# %%
def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    # SOLUTION

    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape
    assert ic == ic2, "in_channels for x and weights don't match up"
    ow = w - kw + 1
    oh = h - kh + 1

    s_b, s_ic, s_h, s_w = x.stride()

    # Get strided x (the new height/width dims have the same stride as the original height/width-strides of x)
    x_new_shape = (b, ic, oh, ow, kh, kw)
    x_new_stride = (s_b, s_ic, s_h, s_w, s_h, s_w)

    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)

    return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)
