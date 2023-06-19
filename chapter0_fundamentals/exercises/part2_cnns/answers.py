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
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"

## EINOPS
# %%

arr = np.load(section_dir / "numbers.npy")

# %%
## EXERCISE 1
array1 = einops.rearrange(arr, "num rgb h w -> rgb h (num w) ")
display_array_as_img(array1)
# %%
## EXERCISE 2
array2 = einops.repeat(arr[0], "rgb h w -> rgb (2 h) w")
display_array_as_img(array2)

# %%
## EXERCISE 3
array3 = einops.repeat(arr[:2], "num rgb h w -> rgb (num h) (2 w)")
display_array_as_img(array3)
# %%
## EXERCISE 4
array4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(array4)
# %%
## EXERCISE 5
array5 = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(array5)
# %%
## EXERCISE 6
array6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2, b2=3)
display_array_as_img(array6)
# %%
## EXERCISE 7
array7 = einops.reduce(arr, "b c h w -> h (b w) ", "max")
display_array_as_img(array7)

# %%
array8 = einops.reduce(arr, "b c h w -> h w", "min")
display_array_as_img(array8)
# %%
array9 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(array9)
# %%
array10 = einops.reduce(
    arr.astype(float), "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", "mean", b1=2
)
display_array_as_img(array10)


# %%
def einsum_trace(mat: np.ndarray):
    # \sum_j A_{ij} B_{jk} -> C_{ik} # A B = C # sum over j
    return einops.einsum(mat, "i i -> ")  # $A_{ij}, tr(A) = \sum_i A_{ii}$
    # A_{ii} -> number # sum over i


tests.test_einsum_trace(einsum_trace)


# %%
def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einops.einsum(mat, vec, "i j, j -> i")


tests.test_einsum_mv(einsum_mv)


# %%
def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einops.einsum(mat1, mat2, "a b, b c -> a c")


tests.test_einsum_mm(einsum_mm)


# %%
def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    return einops.einsum(vec1, vec2, " i, i -> ")


tests.test_einsum_inner(einsum_inner)


# %%
def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    return einops.einsum(vec1, vec2, "i, j -> i j")


tests.test_einsum_outer(einsum_outer)
# %%
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
def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    """
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    """
    # 0 1 2 3 4 i = 0
    # 5 6 7 8 9 i = 1
    # -> 0 6  (i x 5 + i) #mat.shape[1] == 5??
    shape = (mat.shape[0],)
    # asuume square matrix
    stride = (mat.shape[0] + 1,)
    return t.as_strided(mat, shape, stride).sum()


if MAIN:
    tests.test_trace(as_strided_trace)


# %%
mat = t.arange(12).reshape((4, 3))
mat = t.as_strided(mat, (3, 2), (3, 2))  # mat?
vec = t.arange(3)
vec = t.as_strided(vec, (2,), (2,))  #


def as_strided_mv(
    mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]
) -> Float[Tensor, "i"]:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    mat = t.Tensor([[1,1],[0,1],[1,1]]) # shape (3, 2)
    vec = t.Tensor([2,1])  # shape (2,)
    """
    # hi
    # => A, with shape (3, 2) => sum over dim=1 to get shape (3,)
    # you can multiply a (3, 2) with a (3, 2) elementwise like C * D (not "@", which is matmul)
    # (3, 2) * (3, 2) => A
    assert len(mat.shape) == 2
    assert mat.shape[1] == vec.shape[0]
    i, j = mat.shape
    s_vec = vec.stride()[0]
    B = t.as_strided(vec, (i, j), (0, s_vec))
    A = (mat * B).sum(dim=1)
    return A


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)

# %%

matA = t.arange(12).reshape((4, 3))
matB = t.arange(6).reshape((3, 2))
# [1 1]
# [0 1]
# [1 1]
# multiply
# [2 3]
# [1 4]


def as_strided_mm(
    matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]
) -> Float[Tensor, "i k"]:
    """
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    """
    a, b = matA.shape
    c, d = matB.shape
    assert b == c

    A = t.as_strided(matA, (a, b, d), (*matA.stride(), 0))
    B = t.as_strided(matB, (a, c, d), (0, *matB.stride()))
    return (A * B).sum(dim=1)


if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)


# %%
def conv1d_minimal_simple(
    x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]
) -> Float[Tensor, "ow"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments
    left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    """
    width = x.shape[0]
    kernel_width = weights.shape[0]

    expected_output_width = width - kernel_width + 1

    x_stride = x.stride()[0]
    shape = (expected_output_width, kernel_width)
    mat_strides = (x_stride, x_stride)
    mat = t.as_strided(x, shape, mat_strides)
    output = einops.einsum(mat, weights, "i j, j -> i")

    output_width = output.shape[0]
    assert output_width == expected_output_width
    return output


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
    oc, ic, kw = weights.shape
    ow = w - kw + 1

    b_s, ic_s, w_s = x.stride()
    mat = t.as_strided(x, (b, ic, ow, kw), (b_s, ic_s, w_s, w_s))
    output = einops.einsum(mat, weights, "b ic ow kw, oc ic kw -> b oc ow")
    assert output.shape == (b, oc, ow)
    return output


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
    ow = w - kw + 1
    oh = h - kh + 1

    b_s, ic_s, h_s, w_s = x.stride()
    mat = t.as_strided(x, (b, ic, ow, kw, oh, kh), (b_s, ic_s, w_s, w_s, h_s, h_s))
    output = einops.einsum(mat, weights, "b ic ow kw oh kh, oc ic kh kw -> b oc oh ow")
    return output


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)


# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    batch, ic, w = x.shape
    new_x = x.new_full((batch, ic, w + left + right), pad_value)
    new_x[:, :, left : (w + left)] = x
    return new_x


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
    new_shape = (b, ic, top + h + bottom, left + w + right)
    new_x = x.new_full(new_shape, pad_value)
    width_slice = slice(left, w + left)
    height_slice = slice(top, h + top)
    new_x[:, :, height_slice, width_slice] = x
    return new_x


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
    x = pad1d(x, padding, padding, 0)
    b, ic, w = x.shape
    oc, ic, kw = weights.shape
    ow = (w - kw) // stride + 1
    b_s, ic_s, w_s = x.stride()
    mat = t.as_strided(x, (b, ic, ow, kw), (b_s, ic_s, stride * w_s, w_s))
    output = einops.einsum(mat, weights, "b ic ow kw, oc ic kw -> b oc ow")
    assert output.shape == (b, oc, ow)
    return output


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
    padding_h, padding_w = force_pair(padding)
    stride_h, stride_w = force_pair(stride)

    x_padded = pad2d(x, padding_w, padding_w, padding_h, padding_h, 0)
    b, ic, h, w = x_padded.shape
    oc, ic, kh, kw = weights.shape
    ow = (w - kw) // stride_w + 1
    oh = (h - kh) // stride_h + 1
    b_s, ic_s, h_s, w_s = x_padded.stride()

    # mat = t.as_strided(x, (b, ic, ow, kw, oh, kh), (b_s, ic_s, w_s, w_s, h_s, h_s))
    mat = t.as_strided(
        x_padded,
        (b, ic, ow, kw, oh, kh),
        (b_s, ic_s, w_s * stride_w, w_s, h_s * stride_h, h_s),
    )
    output = einops.einsum(mat, weights, "b ic ow kw oh kh, oc ic kh kw -> b oc oh ow")
    assert output.shape == (b, oc, oh, ow)
    return output


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

    kh, kw = force_pair(kernel_size)
    if stride is not None:
        stride_h, stride_w = force_pair(stride)
    else:
        stride_h, stride_w = force_pair(kernel_size)
    padding_h, padding_w = force_pair(padding)

    x_padded = pad2d(x, padding_w, padding_w, padding_h, padding_h, -t.inf)
    b, ic, h, w = x_padded.shape
    oh = (h - kh) // stride_h + 1
    ow = (w - kw) // stride_w + 1
    b_s, ic_s, h_s, w_s = x_padded.stride()

    mat = t.as_strided(
        x_padded,
        (b, ic, oh, kh, ow, kw),
        (b_s, ic_s, h_s * stride_h, h_s, w_s * stride_w, w_s),
    )
    output = t.amax(mat, dim=(3, 5))
    assert output.shape == (b, ic, oh, ow)
    return output


if MAIN:
    tests.test_maxpool2d(maxpool2d)


# %%
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
        getattr(self, "padding")
        result = ""
        for attr in ["kernel_size", "stride", "padding"]:
            value = getattr(self, attr)
            new_result = f"{attr} is {value}\n"
            result += new_result
        return result.strip()


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")


# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:  # max(0, x)
        return t.maximum(x, t.tensor(0))


if MAIN:
    tests.test_relu(ReLU)


# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = tuple(input.shape)
        if self.end_dim < 0:
            end_dim = len(shape) + self.end_dim
        else:
            end_dim = self.end_dim
        if self.start_dim < 0:
            start_dim = len(shape) + self.start_dim
        else:
            start_dim = self.start_dim

        first = shape[:start_dim]
        last = shape[end_dim + 1 :]

        return t.reshape(input, first + (-1,) + last)

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
        dist = t.distributions.Uniform(-1 / in_features, 1 / in_features)
        weights = dist.rsample(t.Size((out_features, in_features)))
        self.weight = nn.Parameter(weights)

        biases = dist.rsample(t.Size((out_features,)))
        if bias:
            self.bias = nn.Parameter(biases)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        bias = self.bias if self.bias is not None else 0
        return (
            einops.einsum(
                x,
                self.weight,
                "... in_features, out_features in_features -> ... out_features",
            )
            + bias
        )

    def extra_repr(self) -> str:
        pass


if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)


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

        kernel_height, kernel_width = force_pair(kernel_size)
        sf = 1 / np.sqrt(in_channels * kernel_width * kernel_height)
        weight = sf * (
            2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1
        )
        self.weight = nn.Parameter(weight)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the functional conv2d you wrote earlier."""
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


if MAIN:
    tests.test_conv2d_module(Conv2d)
