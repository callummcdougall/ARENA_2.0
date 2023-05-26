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

# %%

@dataclass(frozen=True)
class Recipe:
    '''Extra information necessary to run backpropagation. You don't need to modify this.'''

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."

    args: tuple
    "The input arguments passed to func."
    "For instance, if func was np.sum then args would be a length-1 tuple containing the tensor to be summed."

    kwargs: Dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."

    parents: Dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."

# %%

class BackwardFuncLookup:
    def __init__(self) -> None:
        self.lookup = {}

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        self.lookup[(forward_fn, arg_position)] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self.lookup[(forward_fn, arg_position)]


if MAIN:
    BACK_FUNCS = BackwardFuncLookup()
    BACK_FUNCS.add_back_func(np.log, 0, log_back)
    BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
    BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

    assert BACK_FUNCS.get_back_func(np.log, 0) == log_back
    assert BACK_FUNCS.get_back_func(np.multiply, 0) == multiply_back0
    assert BACK_FUNCS.get_back_func(np.multiply, 1) == multiply_back1

    print("Tests passed - BackwardFuncLookup class is working as expected!")

# %%

Arr = np.ndarray

class Tensor:
    '''
    A drop-in replacement for torch.Tensor supporting a subset of features.
    '''

    array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    grad: Optional["Tensor"]
    "Backpropagation will accumulate gradients into this field."
    recipe: Optional[Recipe]
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Union[Arr, list], requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other) -> "Tensor":
        return multiply(other, self)

    def __truediv__(self, other) -> "Tensor":
        return true_divide(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        return true_divide(other, self)

    def __matmul__(self, other) -> "Tensor":
        return matmul(self, other)

    def __rmatmul__(self, other) -> "Tensor":
        return matmul(other, self)

    def __eq__(self, other) -> "Tensor":
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    @property
    def T(self) -> "Tensor":
        return permute(self)

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def expand(self, new_shape):
        return expand(self, new_shape)

    def permute(self, dims):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        '''Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html'''
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
        return bool(self.item())

def empty(*shape: int) -> Tensor:
    '''Like torch.empty.'''
    return Tensor(np.empty(shape))

def zeros(*shape: int) -> Tensor:
    '''Like torch.zeros.'''
    return Tensor(np.zeros(shape))

def arange(start: int, end: int, step=1) -> Tensor:
    '''Like torch.arange(start, end).'''
    return Tensor(np.arange(start, end, step=step))

def tensor(array: Arr, requires_grad=False) -> Tensor:
    '''Like torch.tensor.'''
    return Tensor(array, requires_grad=requires_grad)

# %%

    # array: Arr
    # "The underlying array. Can be shared between multiple Tensors."
    # requires_grad: bool
    # "If True, calling functions or methods on this tensor will track relevant data for backprop."
    # grad: Optional["Tensor"]
    # "Backpropagation will accumulate gradients into this field."
    # recipe: Optional[Recipe]

def log_forward(x: Tensor) -> Tensor:
    '''Performs np.log on a Tensor object.'''
    func = np.log
    args = (x.array,)
    kwargs = dict()
    array = func(*args, **kwargs)
    recipe = None
    requires_grad = grad_tracking_enabled and (x.requires_grad or x.recipe is not None)
    if requires_grad:
        parents = {0: x}
        recipe = Recipe(func=func, args=args, kwargs=kwargs, parents=parents)
    result = Tensor(array=array, requires_grad=requires_grad)
    result.recipe = recipe
    return result


if MAIN:
    log = log_forward
    tests.test_log(Tensor, log_forward)
    tests.test_log_no_grad(Tensor, log_forward)
    a = Tensor([1], requires_grad=True)
    grad_tracking_enabled = False
    b = log_forward(a)
    grad_tracking_enabled = True
    assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
    assert b.recipe is None, "should not create recipe if grad tracking globally disabled"


# %%
def strip_tensor(x: Union[Tensor, Any]) -> Any:
    if isinstance(x, Tensor):
        return x.array
    return x

def multiply_forward(a: Union[Tensor, int], b: Union[Tensor, int]) -> Tensor:
    '''Performs np.log on a Tensor object.'''
    assert isinstance(a, Tensor) or isinstance(b, Tensor)

    func = np.multiply
    args = (strip_tensor(a), strip_tensor(b))
    kwargs = {}
    array = func(*args)
    parents = {i: p for i, p in enumerate([a, b]) if isinstance(p, Tensor)}
    requires_grad = grad_tracking_enabled and any(p.requires_grad or (p.recipe is not None) for p in parents.values())

    out = Tensor(array, requires_grad)

    if requires_grad:
        out.recipe = Recipe(func=func, args=args, kwargs=kwargs, parents=parents)

    return out


if MAIN:
    multiply = multiply_forward
    tests.test_multiply(Tensor, multiply_forward)
    tests.test_multiply_no_grad(Tensor, multiply_forward)
    tests.test_multiply_float(Tensor, multiply_forward)
    a = Tensor([2], requires_grad=True)
    b = Tensor([3], requires_grad=True)
    grad_tracking_enabled = False
    b = multiply_forward(a, b)
    grad_tracking_enabled = True
    assert not b.requires_grad, "should not require grad if grad tracking globally disabled"
    assert b.recipe is None, "should not create recipe if grad tracking globally disabled"