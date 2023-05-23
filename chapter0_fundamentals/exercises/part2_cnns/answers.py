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