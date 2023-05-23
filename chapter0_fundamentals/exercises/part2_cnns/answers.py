# %%

import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
import torch
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
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar  # type: ignore
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img  # type: ignore
import part2_cnns.tests as tests  # type: ignore

MAIN = __name__ == "__main__"

# %%

if MAIN:
    arr = np.load(section_dir / "numbers.npy")
    arr2 = einops.rearrange(arr[:6], "nums c h w -> c h (nums w)")

    display_array_as_img(arr[0])
    display_array_as_img(arr2)

    arr3 = einops.repeat(arr[0], "c h w -> c (2 h) w")
    display_array_as_img(arr3)

    arr4 = einops.repeat(arr[:2], "nums c h w -> c (nums h) (2 w)")
    display_array_as_img(arr4)

    arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
    display_array_as_img(arr5)

    arr6 = einops.rearrange(arr, "(nh nw) c h w -> c (nh h) (nw w)", nw=3)
    display_array_as_img(arr6)

    display_array_as_img(
        einops.reduce(arr.astype(float), "nums c h w -> h (nums w)", "min")
    )

    display_array_as_img(einops.reduce(arr.astype(float), "nums c h w -> h w", "min"))


# %%
def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    return einops.einsum(mat, "i i ->")


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einops.einsum(mat, vec, "i j, j -> i")


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einops.einsum(mat1, mat2, "i j, j k -> i k")


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    return einops.einsum(vec1, vec2, "i, i ->")


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
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
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        dtype=t.float,
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
            output=t.tensor([[0, 1, 2], [5, 6, 7]]),
            size=(2, 3),
            stride=(5, 1),
        ),
        TestCase(
            output=t.tensor([[0, 1, 2], [10, 11, 12]]),
            size=(2, 3),
            stride=(10, 1),
        ),
        TestCase(
            output=t.tensor([[0, 0, 0], [11, 11, 11]]),
            size=(2, 3),
            stride=(11, 0),
        ),
        TestCase(
            output=t.tensor([0, 6, 12, 18]),
            size=(4,),
            stride=(6,),
        ),
    ]

    for i, test_case in enumerate(test_cases):
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
def as_strided_trace(mat: Float[Tensor, "i i"]) -> Float[Tensor, ""]:
    """
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    """
    N = mat.shape[0]
    return mat.as_strided((N,), (N + 1,)).sum()


if MAIN:
    tests.test_trace(as_strided_trace)


# %%
def as_strided_mv(
    mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]
) -> Float[Tensor, "i"]:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    """
    expanded_vec = vec.as_strided(mat.shape, (0, vec.stride(0)))
    return (mat * expanded_vec).sum(dim=1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)


# %%
def as_strided_mm(
    matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]
) -> Float[Tensor, "i k"]:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    """
    common_shape = matA.shape + matB.shape[1:]
    a_rep = matA.as_strided(common_shape, matA.stride() + (0,))
    b_rep = matB.as_strided(common_shape, (0,) + matB.stride())
    return (a_rep * b_rep).sum(dim=1)


if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)

# %%


def conv1d_minimal_simple(
    x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]
) -> Float[Tensor, "ow"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    """
    w = x.shape[0]
    kw = weights.shape[0]
    ow = w - kw + 1
    strided_x = x.as_strided((kw, ow), (x.stride(0), x.stride(0)))
    return einops.einsum(strided_x, weights, "kw ow, kw -> ow")


if MAIN:
    tests.test_conv1d_minimal_simple(conv1d_minimal_simple)


# %%
def conv1d_minimal(
    x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]
) -> Float[Tensor, "b oc ow"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    b, ic, w = x.shape
    oc, ic_, kw = weights.shape
    assert ic == ic_
    ow = w - kw + 1

    new_shape = b, ic, kw, ow
    new_stride = *x.stride(), x.stride(2)
    strided_x = x.as_strided(new_shape, new_stride)
    return einops.einsum(strided_x, weights, "b ic kw ow, oc ic kw -> b oc ow")


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)


# %%
def conv2d_minimal(
    x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]
) -> Float[Tensor, "b oc oh ow"]:
    """
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    b, ic, h, w = x.shape
    oc, ic, kh, kw = weights.shape

    oh = h - kh + 1
    ow = w - kw + 1

    new_shape = b, ic, kh, oh, kw, ow
    new_stride = [x.stride(d) for d in (0, 1, 2, 2, 3, 3)]
    strided_x = x.as_strided(new_shape, new_stride)
    return einops.einsum(
        strided_x, weights, "b ic kh oh kw ow, oc ic kh kw -> b oc oh ow"
    )


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)


# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """

    b, ic, w = x.shape
    out = x.new_full([b, ic, left + right + w], pad_value)
    out[..., left : left + w] = x
    return out


if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)


# %%
def pad2d(
    x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    b, ic, h, w = x.shape
    out = x.new_full([b, ic, top + h + bottom, left + w + right], pad_value)
    out[..., top : top + h, left : left + w] = x
    return out


if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)


# %%
def conv1d(
    x: Float[Tensor, "b ic w"],
    weights: Float[Tensor, "oc ic kw"],
    stride: int = 1,
    padding: int = 0,
) -> Float[Tensor, "b oc ow"]:
    """
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """

    b, ic, w = x.shape
    oc, ic_, kw = weights.shape
    assert ic == ic_
    ow = int((w + 2 * padding - kw) / stride) + 1
    x = pad1d(x, left=padding, right=padding, pad_value=0)

    new_shape = b, ic, kw, ow
    new_stride = *x.stride(), x.stride(2) * stride
    strided_x = x.as_strided(new_shape, new_stride)
    return einops.einsum(strided_x, weights, "b ic kw ow, oc ic kw -> b oc ow")


if MAIN:
    tests.test_conv1d(conv1d)
# %%
IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
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
def conv2d(
    x: Float[Tensor, "b ic h w"],
    weights: Float[Tensor, "oc ic kh kw"],
    stride: IntOrPair = 1,
    padding: IntOrPair = 0,
) -> Float[Tensor, "b oc oh ow"]:
    """
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """

    sh, sw = force_pair(stride)
    ph, pw = force_pair(padding)

    b, ic, h, w = x.shape
    oc, ic, kh, kw = weights.shape

    x = pad2d(x, pw, pw, ph, ph, pad_value=0)

    oh = int((h + 2 * ph - kh) / sh) + 1
    ow = int((w + 2 * pw - kw) / sw) + 1

    new_shape = b, ic, kh, oh, kw, ow
    new_stride = [x.stride(d) for d in (0, 1, 2, 2, 3, 3)]
    new_stride[3] *= sh
    new_stride[5] *= sw
    strided_x = x.as_strided(new_shape, new_stride)
    return einops.einsum(
        strided_x, weights, "b ic kh oh kw ow, oc ic kh kw -> b oc oh ow"
    )


if MAIN:
    tests.test_conv2d(conv2d)


# %%
def maxpool2d(
    x: Float[Tensor, "b ic h w"],
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
) -> Float[Tensor, "b ic oh ow"]:
    """
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    """

    ph, pw = force_pair(padding)
    kh, kw = force_pair(kernel_size)
    sh, sw = force_pair(stride) if stride is not None else (kh, kw)

    b, ic, h, w = x.shape

    x = pad2d(x, pw, pw, ph, ph, pad_value=float("-inf"))

    oh = int((h + 2 * ph - kh) / sh) + 1
    ow = int((w + 2 * pw - kw) / sw) + 1

    new_shape = b, ic, kh, oh, kw, ow
    new_stride = [x.stride(d) for d in (0, 1, 2, 2, 3, 3)]
    new_stride[3] *= sh
    new_stride[5] *= sw
    strided_x = x.as_strided(new_shape, new_stride)

    return einops.reduce(strided_x, "b ic kh oh kw ow -> b ic oh ow", "max")


if MAIN:
    tests.test_maxpool2d(maxpool2d)


# %%
class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = None,
        padding: IntOrPair = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")


# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.maximum(Tensor([0.0]))


if MAIN:
    tests.test_relu(ReLU)


# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        out = x.reshape(
            x.shape[: self.start_dim]
            + (-1,)
            + x.shape[(self.end_dim % len(x.shape)) + 1 :]
        )
        return out

    def extra_repr(self) -> str:
        return ", ".join(
            [f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]]
        )


if MAIN:
    tests.test_flatten(Flatten)
# %%

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.zeros((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        else:
            self.bias = None

        # initialisation
        with torch.no_grad():
            bound = np.sqrt(1 / in_features)
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        out = einops.einsum(self.weight, x, "of if, ... if -> ... of")
        if self.bias is not None:
            out += self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)

    print(Linear(5, 10))


# %%
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, *force_pair(kernel_size))
        )
        k = 1 / np.sqrt(in_channels * torch.prod(Tensor(force_pair(kernel_size))))

        with torch.no_grad():
            self.weight.uniform_(-k, k)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join(
            f"{name}={getattr(self, name)}"
            for name in [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
            ]
        )


if MAIN:
    tests.test_conv2d_module(Conv2d)
# %%
