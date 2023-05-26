#%%
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
# %%
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    '''
    return 1/x * grad_out


if MAIN:
    tests.test_log_back(log_back)
# %%
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    '''
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''
    b_shape = broadcasted.shape
    o_shape = original.shape
    # prepend to o_shape until it's the same length
    o_shape_prepended = (*(1 for _ in range(len(b_shape)-len(o_shape))), *o_shape)
    dims_broadcasted = []
    for dim, (o, b) in enumerate(zip(o_shape_prepended, b_shape)):
        if o != b:
            dims_broadcasted.append(dim)
    result = broadcasted.sum(axis=tuple(dims_broadcasted), keepdims=True)
    return result.reshape(o_shape)


if MAIN:
    tests.test_unbroadcast(unbroadcast)
# %%
