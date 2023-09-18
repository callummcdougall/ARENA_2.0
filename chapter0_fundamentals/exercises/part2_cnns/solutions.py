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
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part2_cnns', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"

# %% 1️⃣ EINOPS AND EINSUM

arr = np.load(section_dir / "numbers.npy")

# %%


if MAIN:
	display_array_as_img(arr[0])

# %%

# FLAT SOLUTION
# Your code here - define arr1
arr1 = einops.rearrange(arr, "b c h w -> c h (b w)")
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr1)

# %%

# FLAT SOLUTION
# Your code here - define arr2
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr2)

# %%

# FLAT SOLUTION
# Your code here - define arr3
arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr3)

# %%

# FLAT SOLUTION
# Your code here - define arr4
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr4)

# %%

# FLAT SOLUTION
# Your code here - define arr5
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr5)

# %%

# FLAT SOLUTION
# Your code here - define arr6
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr6)

# %%

# FLAT SOLUTION
# Your code here - define arr7
arr7 = einops.reduce(arr.astype(float), "b c h w -> h (b w)", "max").astype(int)
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr7)

# %%

# FLAT SOLUTION
# Your code here - define arr8
arr8 = einops.reduce(arr.astype(float), "b c h w -> h w", "min").astype(int)
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr8)

# %%

# FLAT SOLUTION
# Your code here - define arr9
arr9 = einops.rearrange(arr[1], "c h w -> c w h")
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr9)

# %%

# FLAT SOLUTION
# Your code here - define arr10
arr10 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)
# FLAT SOLUTION END

if MAIN:
	display_array_as_img(arr10)

# %%

def einsum_trace(mat: np.ndarray):
	'''
	Returns the same as `np.trace`.
	'''
	return einops.einsum(mat, "i i ->")

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

# %% 2️⃣ ARRAY STRIDES


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
			size=None,
			stride=None,
		),
	
		TestCase(
			output=t.tensor([0, 5, 10, 15]),
			size=None,
			stride=None,
		),
	
		TestCase(
			output=t.tensor([
				[0, 1, 2], 
				[5, 6, 7]
			]), 
			size=None,
			stride=None,
		),
	
		TestCase(
			output=t.tensor([
				[0, 1, 2], 
				[10, 11, 12]
			]), 
			size=None,
			stride=None,
		),
	
		TestCase(
			output=t.tensor([
				[0, 0, 0], 
				[11, 11, 11]
			]), 
			size=None,
			stride=None,
		),
	
		TestCase(
			output=t.tensor([0, 6, 12, 18]), 
			size=None,
			stride=None,
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

	stride = mat.stride()

	assert len(stride) == 2, f"matrix should have size 2"
	assert mat.size(0) == mat.size(1), "matrix should be square"

	diag = mat.as_strided((mat.size(0),), (stride[0] + stride[1],))

	return diag.sum()



if MAIN:
	tests.test_trace(as_strided_trace)

# %%

def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
	'''
	Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
	'''

	sizeM = mat.shape
	sizeV = vec.shape

	strideV = vec.stride()

	assert len(sizeM) == 2, f"mat1 should have size 2"
	assert sizeM[1] == sizeV[0], f"mat{list(sizeM)}, vec{list(sizeV)} not compatible for multiplication"

	vec_expanded = vec.as_strided(mat.shape, (0, strideV[0]))

	product_expanded = mat * vec_expanded

	return product_expanded.sum(dim=1)



if MAIN:
	tests.test_mv(as_strided_mv)
	tests.test_mv2(as_strided_mv)

# %%

def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
	'''
	Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
	'''

	assert len(matA.shape) == 2, f"mat1 should have size 2"
	assert len(matB.shape) == 2, f"mat2 should have size 2"
	assert matA.shape[1] == matB.shape[0], f"mat1{list(matA.shape)}, mat2{list(matB.shape)} not compatible for multiplication"

	# Get the matrix strides, and matrix dims
	sA0, sA1 = matA.stride()
	dA0, dA1 = matA.shape
	sB0, sB1 = matB.stride()
	dB0, dB1 = matB.shape

	expanded_size = (dA0, dA1, dB1)

	matA_expanded_stride = (sA0, sA1, 0)
	matA_expanded = matA.as_strided(expanded_size, matA_expanded_stride)

	matB_expanded_stride = (0, sB0, sB1)
	matB_expanded = matB.as_strided(expanded_size, matB_expanded_stride)

	product_expanded = matA_expanded * matB_expanded

	return product_expanded.sum(dim=1)



if MAIN:
	tests.test_mm(as_strided_mm)
	tests.test_mm2(as_strided_mm)

# %% 3️⃣ CONVOLUTIONS

def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
	'''
	Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

	Simplifications: batch = input channels = output channels = 1.

	x: shape (width,)
	weights: shape (kernel_width,)

	Returns: shape (output_width,)
	'''

	w = x.shape[0]
	kw = weights.shape[0]
	# Get output width, using formula
	ow = w - kw + 1

	# Get strides for x
	s_w = x.stride(0)

	# Get strided x (the new dimension has same stride as the original stride of x)
	x_new_shape = (ow, kw)
	x_new_stride = (s_w, s_w)
	# Common error: s_w is always 1 if the tensor `x` wasn't itself created via striding, so if you put 1 here you won't spot your mistake until you try this with conv2d!
	x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)

	return einops.einsum(x_strided, weights, "ow kw, kw -> ow")



if MAIN:
	tests.test_conv1d_minimal_simple(conv1d_minimal_simple)

# %%

def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
	'''
	Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

	x: shape (batch, in_channels, width)
	weights: shape (out_channels, in_channels, kernel_width)

	Returns: shape (batch, out_channels, output_width)
	'''

	b, ic, w = x.shape
	oc, ic2, kw = weights.shape
	assert ic == ic2, "in_channels for x and weights don't match up"
	# Get output width, using formula
	ow = w - kw + 1

	# Get strides for x
	s_b, s_ic, s_w = x.stride()

	# Get strided x (the new dimension has the same stride as the original width-stride of x)
	x_new_shape = (b, ic, ow, kw)
	x_new_stride = (s_b, s_ic, s_w, s_w)
	# Common error: xsWi is always 1, so if you put 1 here you won't spot your mistake until you try this with conv2d!
	x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)

	return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow",)



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
	assert ic == ic2, "in_channels for x and weights don't match up"
	ow = w - kw + 1
	oh = h - kh + 1

	s_b, s_ic, s_h, s_w = x.stride()

	# Get strided x (the new height/width dims have the same stride as the original height/width-strides of x)
	x_new_shape = (b, ic, oh, ow, kh, kw)
	x_new_stride = (s_b, s_ic, s_h, s_w, s_h, s_w)

	x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)

	return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")



if MAIN:
	tests.test_conv2d_minimal(conv2d_minimal)

# %%

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
	'''Return a new tensor with padding applied to the edges.

	x: shape (batch, in_channels, width), dtype float32

	Return: shape (batch, in_channels, left + right + width)
	'''

	B, C, W = x.shape
	output = x.new_full(size=(B, C, left + W + right), fill_value=pad_value)
	output[..., left : left + W] = x
	# Note - you can't use `left:-right`, because `right` might be zero.
	return output



if MAIN:
	tests.test_pad1d(pad1d)
	tests.test_pad1d_multi_channel(pad1d)

# %%

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
	'''Return a new tensor with padding applied to the edges.

	x: shape (batch, in_channels, height, width), dtype float32

	Return: shape (batch, in_channels, top + height + bottom, left + width + right)
	'''

	B, C, H, W = x.shape
	output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
	output[..., top : top + H, left : left + W] = x
	return output



if MAIN:
	tests.test_pad2d(pad2d)
	tests.test_pad2d_multi_channel(pad2d)

# %%

def conv1d(
	x: Float[Tensor, "b ic w"], 
	weights: Float[Tensor, "oc ic kw"], 
	stride: int = 1, 
	padding: int = 0
) -> Float[Tensor, "b oc ow"]:
	'''
	Like torch's conv1d using bias=False.

	x: shape (batch, in_channels, width)
	weights: shape (out_channels, in_channels, kernel_width)

	Returns: shape (batch, out_channels, output_width)
	'''

	x_padded = pad1d(x, left=padding, right=padding, pad_value=0)

	b, ic, w = x_padded.shape
	oc, ic2, kw = weights.shape
	assert ic == ic2, "in_channels for x and weights don't match up"
	ow = 1 + (w - kw) // stride
	# note, we assume padding is zero in the formula here, because we're working with input which has already been padded

	s_b, s_ic, s_w = x_padded.stride()

	# Get strided x (the new height/width dims have the same stride as the original height/width-strides of x,
	# scaled by the stride (because we're "skipping over" x as we slide the kernel over it))
	# See diagram in hints for more explanation.
	x_new_shape = (b, ic, ow, kw)
	x_new_stride = (s_b, s_ic, s_w * stride, s_w)
	x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)

	return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")



if MAIN:
	tests.test_conv1d(conv1d)

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

def conv2d(
	x: Float[Tensor, "b ic h w"], 
	weights: Float[Tensor, "oc ic kh kw"], 
	stride: IntOrPair = 1, 
	padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
	'''
	Like torch's conv2d using bias=False

	x: shape (batch, in_channels, height, width)
	weights: shape (out_channels, in_channels, kernel_height, kernel_width)

	Returns: shape (batch, out_channels, output_height, output_width)
	'''

	stride_h, stride_w = force_pair(stride)
	padding_h, padding_w = force_pair(padding)

	x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=0)

	b, ic, h, w = x_padded.shape
	oc, ic2, kh, kw = weights.shape
	assert ic == ic2, "in_channels for x and weights don't match up"
	ow = 1 + (w - kw) // stride_w
	oh = 1 + (h - kh) // stride_h

	s_b, s_ic, s_h, s_w = x_padded.stride()

	# Get strided x (new height/width dims have same stride as original height/width-strides of x, scaled by stride)
	x_new_shape = (b, ic, oh, ow, kh, kw)
	x_new_stride = (s_b, s_ic, s_h * stride_h, s_w * stride_w, s_h, s_w)
	x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)

	return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")



if MAIN:
	tests.test_conv2d(conv2d)

# %%

def maxpool2d(
	x: Float[Tensor, "b ic h w"], 
	kernel_size: IntOrPair, 
	stride: Optional[IntOrPair] = None, 
	padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
	'''
	Like PyTorch's maxpool2d.

	x: shape (batch, channels, height, width)
	stride: if None, should be equal to the kernel size

	Return: (batch, channels, output_height, output_width)
	'''

	# Set actual values for stride and padding, using force_pair function
	if stride is None:
		stride = kernel_size
	stride_h, stride_w = force_pair(stride)
	padding_h, padding_w = force_pair(padding)
	kh, kw = force_pair(kernel_size)

	# Get padded version of x
	x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=-t.inf)

	# Calculate output height and width for x
	b, ic, h, w = x_padded.shape
	ow = 1 + (w - kw) // stride_w
	oh = 1 + (h - kh) // stride_h

	# Get strided x
	s_b, s_c, s_h, s_w = x_padded.stride()

	x_new_shape = (b, ic, oh, ow, kh, kw)
	x_new_stride = (s_b, s_c, s_h * stride_h, s_w * stride_w, s_h, s_w)
	x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)

	# Argmax over dimensions of the maxpool kernel
	# (note these are the same dims that we multiply over in 2D convolutions)
	output = t.amax(x_strided, dim=(-1, -2))
	return output



if MAIN:
	tests.test_maxpool2d(maxpool2d)

# %% 4️⃣ MAKING YOUR OWN MODULES

class MaxPool2d(nn.Module):
	def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

	def forward(self, x: t.Tensor) -> t.Tensor:
		'''Call the functional version of maxpool2d.'''
		return maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

	def extra_repr(self) -> str:
		'''Add additional information to the string representation of this class.'''
		return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])



if MAIN:
	tests.test_maxpool2d_module(MaxPool2d)
	m = MaxPool2d(kernel_size=3, stride=2, padding=1)
	print(f"Manually verify that this is an informative repr: {m}")

# %%

class ReLU(nn.Module):
	def forward(self, x: t.Tensor) -> t.Tensor:
		return t.maximum(x, t.tensor(0.0))



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
		self.bias = bias

		sf = 1 / np.sqrt(in_features)

		weight = sf * (2 * t.rand(out_features, in_features) - 1)
		self.weight = nn.Parameter(weight)

		if bias:
			bias = sf * (2 * t.rand(out_features,) - 1)
			self.bias = nn.Parameter(bias)
		else:
			self.bias = None

	def forward(self, x: t.Tensor) -> t.Tensor:
		'''
		x: shape (*, in_features)
		Return: shape (*, out_features)
		'''
		x = einops.einsum(x, self.weight, "... in_feats, out_feats in_feats -> ... out_feats")
		if self.bias is not None:
			x += self.bias
		return x

	def extra_repr(self) -> str:
		# note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
		return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"



if MAIN:
	tests.test_linear_forward(Linear)
	tests.test_linear_parameters(Linear)
	tests.test_linear_no_bias(Linear)

# %%

class Conv2d(nn.Module):
	def __init__(
		self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
	):
		'''
		Same as torch.nn.Conv2d with bias=False.

		Name your weight field `self.weight` for compatibility with the PyTorch version.
		'''
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

		kernel_height, kernel_width = force_pair(kernel_size)
		sf = 1 / np.sqrt(in_channels * kernel_width * kernel_height)
		weight = sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1)
		self.weight = nn.Parameter(weight)

	def forward(self, x: t.Tensor) -> t.Tensor:
		'''Apply the functional conv2d you wrote earlier.'''
		return conv2d(x, self.weight, self.stride, self.padding)

	def extra_repr(self) -> str:
		keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
		return ", ".join([f"{key}={getattr(self, key)}" for key in keys])



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
		x=range(10),
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
	probs = model(img_input).squeeze().softmax(-1).detach()
	
	bar(
		probs,
		x=range(10),
		template="ggplot2",
		width=600,
		title="Classification probabilities", 
		labels={"x": "Digit", "y": "Probability"}, 
		text_auto='.2f',
		showlegend=False, 
		xaxis_tickmode="linear"
	)

# %%

