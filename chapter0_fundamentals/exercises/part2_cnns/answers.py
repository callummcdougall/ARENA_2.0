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
    x_stride = (x.stride()[-4],
                0,
                x.stride()[-3],
                x.stride()[-2],
                x.stride()[-2],
                x.stride()[-1],
                x.stride()[-1])
    k_stride = (0,
                weights.stride()[-4],
                weights.stride()[-3],
                0,
                weights.stride()[-2],
                0,
                weights.stride()[-1])

    x_part = x.as_strided(size=x_size, stride=x_stride)
    k_part = weights.as_strided(size=x_size, stride=k_stride)
    return einops.einsum(x_part*k_part, "b oc ic oh kh ow kw -> b oc oh ow")

if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)

# %%

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    results = x.new_full((*x.shape[:-1], x.shape[-1] + left + right), pad_value)
    results[..., left:left + x.shape[-1]] = x
    return results

if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)


# %%

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    results = x.new_full((*x.shape[:-2], 
                          x.shape[-2] + top + bottom,
                          x.shape[-1] + left + right),
                          pad_value)
    results[..., top:top + x.shape[-2], left:left + x.shape[-1]] = x
    return results


if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)

# %%

IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

from solutions import maxpool2d

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        # SOLUTION
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        # SOLUTION
        return maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        # SOLUTION
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")

# %%

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x * ( x > 0)


if MAIN:
    tests.test_relu(ReLU)

# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''

        shape = input.shape

        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        shape_left = shape[:start_dim]
        # shape_middle = t.prod(t.tensor(shape[start_dim : end_dim+1])).item()
        shape_middle = functools.reduce(lambda x, y: x*y, shape[start_dim : end_dim+1])
        shape_right = shape[end_dim+1:]

        new_shape = shape_left + (shape_middle,) + shape_right

        return t.reshape(input, new_shape)
        

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])



if MAIN:
    print(Flatten(0, -1).forward(t.tensor([[1,2,3],[2,3,4]])))
    tests.test_flatten(Flatten)

# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = t.rand(out_features, in_features)
        self.weight = (self.weight*2 - 1)/t.tensor([in_features]).sqrt()
        self.weight = nn.Parameter(self.weight)
        self.bias = None
        if bias:
            self.bias = t.rand(out_features)
            self.bias = (self.bias*2 - 1)/t.tensor([in_features]).sqrt()
            self.bias = nn.Parameter(self.bias)
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        mul = t.mm(x,self.weight.T)
        if self.bias is not None:
            mul += self.bias
        return mul

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)

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
from solutions import conv2d

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        self.weight = t.rand(out_channels, in_channels,*force_pair(kernel_size))
        N_k = in_channels*force_pair(kernel_size)[0]*force_pair(kernel_size)[1]
        self.weight = (self.weight*2 - 1)/t.tensor([N_k]).sqrt()
        self.weight = nn.Parameter(self.weight)
        self.stride = force_pair(stride)
        self.padding = force_pair(padding)
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        pass


if MAIN:
    tests.test_conv2d_module(Conv2d)


# %%

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(in_features=32*14*14, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.fc(self.flatten(self.relu(self.maxpool(self.conv(x)))))



if MAIN:
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    print(model)

# %%
if MAIN:
    MNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset



if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=True)

# %%

if MAIN:
    img, label = mnist_trainset[1]

    imshow(
        img.squeeze(), 
        color_continuous_scale="gray", 
        zmin=img.min().item(),
        zmax=img.max().item(),
        title=f"Digit = {label}",
        width=450,
    )
# %%
if MAIN:
    img_input = img.unsqueeze(0).to(device) # add batch dimension
    probs = model(img_input).squeeze().softmax(-1).detach()

    bar(
        probs,
        x=range(1, 11),
        template="ggplot2",
        width=600,
        title="Classification probabilities", 
        labels={"x": "Digit", "y": "Probability"}, 
        text_auto='.2f',
        showlegend=False, 
        xaxis_tickmode="linear"
    )

# %%

if MAIN:
    batch_size = 64
    epochs = 3

    mnist_trainset, _ = get_mnist(subset = 10)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)

    optimizer = t.optim.Adam(model.parameters())
    loss_list = []

    for epoch in tqdm(range(epochs)):
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())   # .item() converts single-elem tensor to scalar


# %%

if MAIN:
    line(
        loss_list, 
        yaxis_range=[0, max(loss_list) + 0.1],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"}, 
        title="ConvNet training on MNIST (cross entropy loss)",
        width=700
    )

# %%

if MAIN:
    img = mnist_trainset[99][0]
    probs = model(img.unsqueeze(0).to(device)).squeeze().softmax(-1).detach()
    imshow(
        img.squeeze(), 
        color_continuous_scale="gray", 
        zmin=img.min().item(),
        zmax=img.max().item(),
        title=f"Digit = {label}",
        width=450,
    )
    bar(
        probs,
        x=range(1, 11),
        template="ggplot2",
        width=600,
        title="Classification probabilities", 
        labels={"x": "Digit", "y": "Probability"}, 
        text_auto='.2f',
        showlegend=False, 
        xaxis_tickmode="linear"
    )
# %%
