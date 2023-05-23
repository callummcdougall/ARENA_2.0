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
