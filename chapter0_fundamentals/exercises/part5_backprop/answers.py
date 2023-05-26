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

def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    '''Backwards function for x * y wrt argument 0 aka x.'''
    if not isinstance(y, Arr):
        y = np.array(y)
    return unbroadcast(y * grad_out, x)
    

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
    # forward pass
    d = a*b
    e = np.log(c)
    f = d*e
    g = np.log(f)

    # backward pass
    dg_df = log_back(np.ones_like(g), g, f)
    dg_dd = multiply_back0(dg_df, f, d, e)
    dg_de = multiply_back1(dg_df, f, d, e)
    dg_dc = log_back(dg_de, e ,c)
    dg_db = multiply_back1(dg_dd, d, a, b)
    dg_da = multiply_back0(dg_dd, d, a, b)

    return dg_da, dg_db, dg_dc



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
    funcs: Dict[Tuple[Callable, int], Callable]

    def __init__(self) -> None:
        self.funcs = dict()

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        self.funcs[(forward_fn, arg_position)] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self.funcs[(forward_fn, arg_position)]


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
def log_forward(x: Tensor) -> Tensor:
    '''Performs np.log on a Tensor object.'''
    should_grad = grad_tracking_enabled and (x.requires_grad or x.recipe is not None)

    array = np.log(x.array)
    result = tensor(array, requires_grad=should_grad)
    if should_grad:
        result.recipe = Recipe(func=np.log, args=(x.array,),
                               kwargs=dict(), parents=(x,))
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

#%%
def multiply_forward(a: Union[Tensor, int], b: Union[Tensor, int]) -> Tensor:
    '''Performs np.log on a Tensor object.'''
    assert isinstance(a, Tensor) or isinstance(b, Tensor)
    
    args = []
    parents = dict()
    a_requires_grad = None
    b_requires_grad = None
    if isinstance(a, Tensor):
        parents[0] = a
        args.append(a.array)
        a_requires_grad = (a.requires_grad or a.recipe is not None)
    else:
        args.append(a)
        a_requires_grad = False

    if isinstance(b, Tensor):
        parents[1] = b
        args.append(b.array)
        b_requires_grad = (b.requires_grad or b.recipe is not None)
    else:
        args.append(b)
        b_requires_grad = False


    should_grad = grad_tracking_enabled and (a_requires_grad or b_requires_grad)

    array = np.multiply(*args)
    result = tensor(array, requires_grad=should_grad)
    if should_grad:
        result.recipe = Recipe(func=np.multiply, args=tuple(args),
                               kwargs=dict(), parents=parents)
    return result
    


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
    assert b.recipe is None, "should not create recipe if grad tracking globally disabled"# %%

#%%
def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    '''
    numpy_func: Callable
        takes any number of positional arguments, some of which may be NumPy arrays, and 
        any number of keyword arguments which we aren't allowing to be NumPy arrays at 
        present. It returns a single NumPy array.

    is_differentiable: 
        if True, numpy_func is differentiable with respect to some input argument, so we 
        may need to track information in a Recipe. If False, we definitely don't need to
        track information.

    Return: Callable
        It has the same signature as numpy_func, except wherever there was a NumPy array, 
        this has a Tensor instead.
    '''

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        recipe_args = []
        parents = dict()
        array_requires_grad = False

        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                parents[i] = arg
                recipe_args.append(arg.array)
                if (arg.requires_grad or arg.recipe is not None):
                    array_requires_grad =  True 
            else:
                recipe_args.append(arg)


        should_grad = grad_tracking_enabled and array_requires_grad and is_differentiable

        array = numpy_func(*recipe_args, **kwargs)
        result = tensor(array, requires_grad=should_grad)
        if should_grad:
            result.recipe = Recipe(func=numpy_func, args=tuple(recipe_args),
                                kwargs=kwargs, parents=parents)
        return result

    return tensor_func
    

def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    # need to be careful with sum, because kwargs have different names in torch and numpy
    return np.sum(x, axis=dim, keepdims=keepdim)


if MAIN:
    log = wrap_forward_fn(np.log)
    multiply = wrap_forward_fn(np.multiply)
    eq = wrap_forward_fn(np.equal, is_differentiable=False)
    sum = wrap_forward_fn(_sum)

    tests.test_log(Tensor, log)
    tests.test_log_no_grad(Tensor, log)
    tests.test_multiply(Tensor, multiply)
    tests.test_multiply_no_grad(Tensor, multiply)
    tests.test_multiply_float(Tensor, multiply)
    tests.test_sum(Tensor)
# %%
class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> List[Node]:
    return node.children


def topological_sort(node: Node, get_children: Callable) -> List[Node]:
    '''
    Return a list of node's descendants in reverse topological order from future to past (i.e. `node` should be last).

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    '''
    # SOLUTION

    result: List[Node] = [] # stores the list of nodes to be returned (in reverse topological order)
    perm: set[Node] = set() # same as `result`, but as a set (faster to check for membership)
    temp: set[Node] = set() # keeps track of previously visited nodes (to detect cyclicity)

    def visit(cur: Node):
        '''
        Recursive function which visits all the children of the current node, and appends them all
        to `result` in the order they were found.
        '''
        if cur in perm:
            return
        if cur in temp:
            raise ValueError("Not a DAG!")
        temp.add(cur)

        for next in get_children(cur):
            visit(next)

        result.append(cur)
        perm.add(cur)
        temp.remove(cur)

    visit(node)
    return result


if MAIN:
    tests.test_topological_sort_linked_list(topological_sort)
    tests.test_topological_sort_branching(topological_sort)
    tests.test_topological_sort_rejoining(topological_sort)
    tests.test_topological_sort_cyclic(topological_sort)



def get_parents(tensor: Tensor):
    if tensor.recipe is not None:
        return tensor.recipe.parents.values()
    return []
#%%

def sorted_computational_graph(tensor: Tensor) -> List[Tensor]:
    '''
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph, 
    in reverse topological order (i.e. `tensor` should be first).
    '''
    return reversed(topological_sort(tensor, get_children=get_parents))



if MAIN:
    a = Tensor([1], requires_grad=True)
    b = Tensor([2], requires_grad=True)
    c = Tensor([3], requires_grad=True)
    d = a * b
    e = c.log()
    f = d * e
    g = f.log()
    name_lookup = {a: "a", b: "b", c: "c", d: "d", e: "e", f: "f", g: "g"}

    print([name_lookup[t] for t in sorted_computational_graph(g)])
# %%

def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    '''Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node: 
        The rightmost node in the computation graph. 
        If it contains more than one element, end_grad must be provided.
    end_grad: 
        A tensor of the same shape as end_node. 
        Set to 1 if not specified and end_node has only one element.
    '''
    order = sorted_computational_graph(end_node)
    grads: Dict[Tensor, Arr] = dict()

    for i, node in enumerate(order):
        if i == 0: # initialize the root node
            if end_grad is not None:
                grads[node] = end_grad.array.astype('float64')
            else:
                grads[node] = np.ones_like(node.array, dtype='float64')
        elif node.requires_grad and node.is_leaf: 
            # compute and store the total gradient of the node if they are leaves and require grad
            if node.grad is None:
                node.grad = tensor(grads[node])
            else:
                node.grad.array += grads[node]


        # compute gradients of all parents using the value of this node
        if node.recipe is not None:
            for position, parent in node.recipe.parents.items():
                backward_f = BACK_FUNCS.get_back_func(node.recipe.func, position)
                if parent not in grads:
                    grads[parent] = np.zeros_like(node.array, dtype='float64')
                grads[parent] += backward_f(grads[node], node.array,
                                            *node.recipe.args, **node.recipe.kwargs)
    

if MAIN:
    tests.test_backprop(Tensor)
    tests.test_backprop_branching(Tensor)
    tests.test_backprop_requires_grad_false(Tensor)
    tests.test_backprop_float_arg(Tensor)
# %%
# def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
#     '''Backward function for f(x) = -x elementwise.'''
#     #return (-1 * grad_out).reshape(x.shape)
#     return np.full_like(x, -1) * grad_out

def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backward function for f(x) = -x elementwise.'''
    # SOLUTION
    return unbroadcast(-grad_out, x)


if MAIN:
    negative = wrap_forward_fn(np.negative)
    BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

    tests.test_negative_back(Tensor)
#%%
def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    '''Like torch.add_. Compute x += other * alpha in-place and return tensor.'''
    np.add(x.array, other.array * alpha, out=x.array)
    return x


def safe_example():
    '''This example should work properly.'''
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    a.add_(b)
    c = a * b
    c.sum().backward()
    assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
    assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


def unsafe_example():
    '''This example is expected to compute the wrong gradients.'''
    a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
    c = a * b
    a.add_(b)
    c.sum().backward()
    if a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0]):
        print("Grad wrt a is OK!")
    else:
        print("Grad wrt a is WRONG!")
    if b.grad is not None and np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0]):
        print("Grad wrt b is OK!")
    else:
        print("Grad wrt b is WRONG!")



if MAIN:
    safe_example()
    unsafe_example()


# %%

class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        '''Share the array with the provided tensor.'''
        super().__init__(tensor.array)
        self.requires_grad = requires_grad

    def __repr__(self):
        return ""


if MAIN:
    x = Tensor([1.0, 2.0, 3.0])
    p = Parameter(x)
    assert p.requires_grad
    assert p.array is x.array
    assert repr(p) == ""
    x.add_(Tensor(np.array(2.0)))
    assert np.allclose(
        p.array, np.array([3.0, 4.0, 5.0])
    ), "in-place modifications to the original tensor should affect the parameter"

# %%
class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        '''Share the array with the provided tensor.'''
        # SOLUTION
        return super().__init__(tensor.array, requires_grad=requires_grad)

    def __repr__(self):
        # SOLUTION
        return f"Parameter containing:\n{super().__repr__()}"

class Module:
    _modules: Dict[str, "Module"]
    _parameters: Dict[str, Parameter]

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self):
        '''Return the direct child modules of this module.'''
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        '''
        Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        '''
        # SOLUTION
        parameters_list = list(self.__dict__["_parameters"].values())
        if recurse:
            for mod in self.modules():
                parameters_list.extend(list(mod.parameters(recurse=True)))
        return iter(parameters_list)

    def __setattr__(self, key: str, val: Any) -> None:
        '''
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call __setattr__ from the superclass.
        '''
        # SOLUTION
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Union[Parameter, "Module"]:
        '''
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        '''
        # SOLUTION
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        raise KeyError(key)


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        def _indent(s_, numSpaces):
            return re.sub("\n", "\n" + (" " * numSpaces), s_)
        lines = [f"({key}): {_indent(repr(module), 2)}" for key, module in self._modules.items()]
        return "".join([
            self.__class__.__name__ + "(",
            "\n  " + "\n  ".join(lines) + "\n" if lines else "", ")"
        ])
#%%

def _matmul2d(x: Arr, y: Arr) -> Arr:
    '''Matrix multiply restricted to the case where both inputs are exactly 2D.'''
    return x @ y

def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    pass

def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    pass


if MAIN:
    matmul = wrap_forward_fn(_matmul2d)
    BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
    BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)

    tests.test_matmul2d(Tensor)


# %%


class Linear(Module):
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.weight = Parameter(tensor(np.random.uniform(size=(out_features, in_features))))
        self.bias = Parameter(tensor(np.random.uniform(size=(out_features,))))

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        xw = np.matmul(self.weight.array, x.array)
        return tensor(xw + self.bias.array, requires_grad=True)

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"



if MAIN:
    linear = Linear(3, 4)
    assert isinstance(linear.weight, Tensor)
    assert linear.weight.requires_grad

    input = Tensor([1.0, 2.0, 3.0])
    output = linear(input)
    assert output.requires_grad

    expected_output = linear.weight.array @ input.array + linear.bias
    np.testing.assert_allclose(output.array, expected_output.array)

    print("All tests for `Linear` passed!")
# %%

class NoGrad:
    '''Context manager that disables grad inside the block. Like torch.no_grad.'''

    was_enabled: bool

    def __enter__(self):
        '''
        Method which is called whenever the context manager is entered, i.e. at the 
        start of the `with NoGrad():` block.
        '''
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False
        

    def __exit__(self, type, value, traceback):
        '''
        Method which is called whenever we exit the context manager.
        '''
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled 

#%

def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    '''Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    '''
    pass


if MAIN:
    tests.test_cross_entropy(Tensor, cross_entropy)