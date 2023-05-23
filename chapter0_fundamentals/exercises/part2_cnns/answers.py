# %% Imports

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
arr1 = arr[0]

if MAIN:
    display_array_as_img(arr1)

# %%
print(arr.shape)

arr2 = einops.rearrange(arr, 'n c w h -> c w (n h)')

print(arr2.shape)

if MAIN:
    display_array_as_img(arr2)
# %%

# Your code here - define arr3
arr3 = einops.repeat(arr[0], 'c h w -> c (n h) w', n=2)

if MAIN:
    display_array_as_img(arr3)




# %%
arr4 = einops.rearrange(arr[:2], 'n c h w -> c (n h) w')
arr4 = einops.repeat(arr4, 'c h w -> c h (n w)', n=2 )

if MAIN:
    display_array_as_img(arr4)
# %%
arr5 = einops.repeat(arr[0], 'c h w -> c (h n) w', n=2)

if MAIN:
    display_array_as_img(arr5)
# %%

arr6 = einops.rearrange(arr[0], 'c h w -> h (c w) ')

if MAIN:
    display_array_as_img(arr6)

# %%

arr7 = einops.reduce(arr, 'n c h w -> h (n w)', 'max')

if MAIN:
    display_array_as_img(arr7)
# %%

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    print(f'input: {mat}')
    mat = einops.einsum(mat, 'i i ->')
    print(f'our mat: {mat}')
    return mat

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    result = einops.einsum(mat, vec, 'i j, j -> i')
    return result

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    result = einops.einsum(mat1, mat2, 'i j, j k -> i k')
    return result

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    result = einops.einsum(vec1, vec2, 'i, i -> ')
    return result

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    result = einops.einsum(vec1, vec2, 'i, j -> i j')
    return result


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
            size=(2,3),
            stride=(5,1),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2, 3),
            stride=(10,1),
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2,3),
            stride=(11,0),
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
if MAIN:
    test_input2 = t.arange(25)
    test_input2 = einops.rearrange(test_input2, "(a b) -> a b", a=5)
    print(test_input2)

# %%
def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    mat_view = mat.as_strided(
        size=(mat.shape[0],),
        stride=(mat.shape[0] +1,)
    )
    return(t.sum(mat_view))

as_strided_trace(test_input2)
#%%
if MAIN:
    tests.test_trace(as_strided_trace)

# %%
test_input2.stride()


#%%
test_vec = test_input2[0]




# %%
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    stride = vec.stride()
    
    vec_rep = vec.as_strided(
        size=(mat.shape),
        stride=(0,stride[-1])
    )
    print(f'vec {vec}')
    print(f'mat {mat}')
    print(f'vec_rep {vec_rep}')
    print(f'stride {stride}')
    return t.sum(mat * vec_rep, -1)

as_strided_mv(test_input2, test_vec)


# %%

a = t.tensor([[1., 2.], [3., 4.]])

print(a.as_strided(size=(2,2,2), stride=(1,1,1)))

#%%

if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)

# %%
def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    
    matA_exp = matA.as_strided(
        size=(),
        stride=((matA.stride(0), matA.stride(1), 0))
    )
    
    matB_exp = matB.as_strided(
        size=(),
        stride=((matB.stride(0), matB.stride(1), 0))
    )

    pass


if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)


#%% Convolutions

def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    # SOLUTION

    w = x.shape[0]
    kw = weights.shape[0]
    # Get output width, using formula
    ow = w - kw + 1

    # Get strides for x
    s_w = x.stride(0)

    # Get strided x (the new dimension has same stride as the original stride of x)
    x_new_shape = (ow, kw)
    x_new_stride = (s_w, s_w)
    # Common error: s_w is always 1 if the tensor `x` wasn't itself created via striding, so if you put 1 here you won't spot your mistake until you try this with conv2d!
    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)

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
    # SOLUTION

    b, ic, w = x.shape
    oc, ic2, kw = weights.shape
    assert ic == ic2, "in_channels for x and weights don't match up"
    # Get output width, using formula
    ow = w - kw + 1

    # Get strides for x
    s_b, s_ic, s_w = x.stride()

    # Get strided x (the new dimension has the same stride as the original width-stride of x)
    x_new_shape = (b, ic, ow, kw)
    x_new_stride = (s_b, s_ic, s_w, s_w)
    # Common error: xsWi is always 1, so if you put 1 here you won't spot your mistake until you try this with conv2d!
    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)

    return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow",)


# %%
def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    pass

    # old shapes
    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape
    assert ic == ic2

    # new shape
    h_new = h - (kh -1)
    w_new = w - (kw -1)

    # old stride
    s_b, s_ic, s_h, s_w = x.stride()

    # new stride
    x_new_shape = (b, ic, h_new, w_new, kh, kw)
    x_new_stride = (s_b, s_ic, s_h, s_w, s_h, s_w)

    # apply new stride
    x_strided = x.as_strided(
        size=x_new_shape,
        stride=x_new_stride
    )

    return einops.einsum(x_strided, weights, "b ic h_new w_new kh kw, oc ic kh kw -> b oc h_new w_new")



if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)
# %%
IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        self.maxpool2d(x, self.kernel_size, self.stride, self.padding)
        pass

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        pass

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
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")