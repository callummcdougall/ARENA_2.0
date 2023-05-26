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
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

import part5_backprop.tests as tests
from part5_backprop.utils import visualize, get_mnist
from plotly_utils import line

MAIN = __name__ == "__main__"

# 01 - Introduction

def multiply_back(grad_out, out, a, b):
    '''
    Inputs:
        grad_out = dL/d(out)
        out = a * b

    Returns:
        dL/da
    '''
    return grad_out * b


# %% Exercie Log-back
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
# %% Unbroadcast
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    '''
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''
    b_shape = broadcasted.shape 
    o_shape = original.shape
    dims_to_sum = len(b_shape) - len(o_shape)
    # Case Same size 
    if len(b_shape) != len(o_shape):
        indices = tuple(i for i in range(dims_to_sum))
        broadcasted =  broadcasted.sum(axis=indices, keepdims=False)
        b_shape = broadcasted.shape
    
    # Find out if some dims got broadcasted
    dims_to_sum = tuple([
        i for i, (o, b) in enumerate(zip(original.shape, broadcasted.shape))
        if o == 1 and b > 1
    ])    
    return broadcasted.sum(axis= dims_to_sum, keepdims=True)

if MAIN:
    tests.test_unbroadcast(unbroadcast)

# %% multily back function
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    '''Backwards function for x * y wrt argument 0 aka x.'''
    if not isinstance(y, Arr):
        y = np.array(y)
    # 1. Calculate the derivative wrt the unbroadcasted version
    derivative = grad_out * y
    # 2. Use unbroadcast to get the derivative wrt the original version
    return unbroadcast(derivative, x)

def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    '''Backwards function for x * y wrt argument 1 aka y.'''
    if not isinstance(x, Arr):
        x = np.array(x)
    # 1. Calculate the derivative wrt the unbroadcasted version
    derivative = grad_out * x
    # 2. Use unbroadcast to get the derivative wrt the original version
    return unbroadcast(derivative, y)


if MAIN:
    tests.test_multiply_back(multiply_back0, multiply_back1)
    tests.test_multiply_back_float(multiply_back0, multiply_back1)

# %% Implement forward & backward
def forward_and_back(a: Arr, b: Arr, c: Arr) -> Tuple[Arr, Arr, Arr]:
    '''
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    '''
    
    # Step 1: Calculate output g 
    d = a * b
    e = np.log(c)
    f = d * e 
    g = np.log(f)

    # Step 2: Backprop gradients
    dg_df = log_back(1, out=g, x=f)
    dg_de = multiply_back1(grad_out=dg_df, out= f,
                        x=d,y=e)
    dg_dd = multiply_back0(grad_out=dg_df,out=f,
                        x=d,y=e)
    dg_da =  multiply_back0(grad_out=dg_dd, out=d,
                        x=a,y=b)
    dg_db =  multiply_back1(grad_out=dg_dd, out=d,
                        x=a,y=b)
    dg_dc = log_back(dg_de, out=e, x=c)

    return dg_da, dg_db, dg_dc 


if MAIN:
    tests.test_forward_and_back(forward_and_back)

# SECTION 2: AUTOGRAD
# %%
