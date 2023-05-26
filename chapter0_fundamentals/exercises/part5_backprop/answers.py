# %%
import os
import sys
import re
import time
import torch as t
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Iterable, Optional, Union, Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_backprop"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

import part5_backprop.tests as tests
from part5_backprop.utils import visualize, get_mnist
from plotly_utils import line

MAIN = __name__ == "__main__"

# %%


def multiply_back(grad_out, out, a, b):
    """
    Inputs:
        grad_out = dL/d(out)
        out = a * b

    Returns:
        dL/da
    """
    return grad_out * b


# %%

def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    '''
    return grad_out / x


if MAIN:
    tests.test_log_back(log_back)

# %%
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    '''
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''
    shape_len_diff = len(broadcasted.shape) - len(original.shape)
    if shape_len_diff > 0:
        ub = broadcasted.sum(axis=tuple(range(shape_len_diff)))
    else:
        ub = broadcasted

    dimsum = []
    for i, (ub_dimsize, og_dimsize) in enumerate(zip(ub.shape, original.shape)):
        if ub_dimsize != og_dimsize:
            dimsum.append(i)

    return ub.sum(axis=tuple(dimsum), keepdims=True)


if MAIN:
    tests.test_unbroadcast(unbroadcast)

# %%
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    '''Backwards function for x * y wrt argument 0 aka x.'''
    if not isinstance(y, Arr):
        y = np.array(y)
    return unbroadcast(grad_out * y, x)

def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    '''Backwards function for x * y wrt argument 1 aka y.'''
    if not isinstance(x, Arr):
        x = np.array(x)
    return unbroadcast(grad_out * x, y)


if MAIN:
    tests.test_multiply_back(multiply_back0, multiply_back1)
    tests.test_multiply_back_float(multiply_back0, multiply_back1)

# %%

def forward_and_back(a: Arr, b: Arr, c: Arr) -> Tuple[Arr, Arr, Arr]:
    '''
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    '''
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)

    df  = log_back(np.array(1), g, f)
    
    dd = multiply_back0(df, f, d, e)
    de = multiply_back1(df, f, d, e)

    da = multiply_back0(dd, d, a, b)
    db = multiply_back1(dd, d, a, b)
    
    dc = log_back(de, e, c)

    return da, db, dc

if MAIN:
    tests.test_forward_and_back(forward_and_back)