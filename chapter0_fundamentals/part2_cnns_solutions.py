# %%

from fancy_einsum import einsum
from typing import Union, Optional, Tuple
import numpy as np
import torch as t
from torch.nn import functional as F
from collections import namedtuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from IPython.display import display
import plotly.express as px
from PIL import Image
import functools

from part2_cnns_utils import display_array_as_img
import part2_cnns_tests as tests

MAIN = __name__ == "__main__"





# %% SECTION 1: EINOPS AND EINSUM

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einsum("i i", mat)

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einsum("i j, j -> i", mat, vec)

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einsum("i j, j k -> i k", mat1, mat2)

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einsum("i, i", vec1, vec2)

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einsum("i, j -> i j", vec1, vec2)






# %% SECTION 2: ARRAY STRIDES

test_input = t.tensor(
    [[0, 1, 2, 3, 4], 
    [5, 6, 7, 8, 9], 
    [10, 11, 12, 13, 14], 
    [15, 16, 17, 18, 19]], dtype=t.float
)


TestCase = namedtuple("TestCase", ["output", "size", "stride"])

test_cases = [
    TestCase(
        output=t.tensor([0, 1, 2, 3]), 
        size=(4,),
        stride=(1,)
    ),
    # Explanation: the output is a 1D vector of length 4 (hence size=(4,))
    # and each time you move one element along in this output vector, you also want to move
    # one element along the `test_input_a` tensor

    TestCase(
        output=t.tensor([0, 1, 2, 3, 4]),
        size=(5,),
        stride=(1,)
    ),
    # Explanation: the tensor is held in a contiguous memory block. When you get to the end
    # of one row, a single stride jumps to the start of the next row

    TestCase(
        output=t.tensor([0, 5, 10, 15]),
        size=(4,),
        stride=(5,)
    ),
    # Explanation: this is same as previous case, only now you're moving in colspace (i.e. skipping
    # 5 elements) each time you move one element across the output tensor.
    # So stride is 5 rather than 1

    TestCase(
        output=t.tensor([
            [0, 1, 2], 
            [5, 6, 7]
        ]), 
        size=(2, 3),
        stride=(5, 1)
    ),
    # Explanation: consider the output tensor. As you move one element along a row, you want to jump
    # one element in the `test_input_a` (since you're just going to the next row). As you move
    # one element along a column, you want to jump to the next column, i.e. a stride of 5.

    TestCase(
        output=t.tensor([
            [0, 1, 2], 
            [10, 11, 12]
        ]), 
        size=(2, 3),
        stride=(10, 1)
    ),

    TestCase(
        output=t.tensor([
            [0, 0, 0], 
            [11, 11, 11]
        ]), 
        size=(2, 3),
        stride=(11, 0)
    ),

    TestCase(
        output=t.tensor([0, 6, 12, 18]), 
        size=(4,),
        stride=(6,)
    ),

    TestCase(
        output=t.tensor([
            [[0, 1, 2]], 
            [[9, 10, 11]]
        ]), 
        size=(2, 1, 3),
        stride=(9, 0, 1)
    ),
    # Note here that the middle element of `stride` doesn't actually matter, since you never
    # jump in this dimension. You could change it and the test result would still be the same

    TestCase(
        output=t.tensor(
            [
                [
                    [[0, 1], [2, 3]],
                    [[4, 5], [6, 7]]
                ], 
                [
                    [[12, 13], [14, 15]], 
                    [[16, 17], [18, 19]]
                ]
            ]
        ),
        size=(2, 2, 2, 2),
        stride=(12, 4, 2, 1)
    ),
]
if MAIN:
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

def as_strided_trace(mat: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    
    stride = mat.stride()
    
    assert len(stride) == 2, f"matrix should have size 2"
    assert mat.size(0) == mat.size(1), "matrix should be square"
    
    return mat.as_strided((mat.size(0),), (sum(stride),)).sum()


def as_strided_mv(mat: t.Tensor, vec: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    
    sizeM = mat.shape
    sizeV = vec.shape
    
    strideM = mat.stride()
    strideV = vec.stride()
    
    assert len(sizeM) == 2, f"mat1 should have size 2"
    assert sizeM[1] == sizeV[0], f"mat{list(sizeM)}, vec{list(sizeV)} not compatible for multiplication"
    
    vec_expanded = vec.as_strided(mat.shape, (0, strideV[0]))
    
    product_expanded = mat * vec_expanded
    
    return product_expanded.sum(dim=1)


def as_strided_mm(matA: t.Tensor, matB: t.Tensor) -> t.Tensor:
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
    tests.test_trace(as_strided_trace)
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)






# %% SECTION 3: CONVOLUTIONS

def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    
    batch, in_channels, width = x.shape
    out_channels, in_channels_2, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = width - kernel_width + 1
    
    xsB, xsI, xsWi = x.stride()
    wsO, wsI, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_width, kernel_width)
    x_new_stride = (xsB, xsI, xsWi, xsWi)
    # Common error: xsWi is always 1, so if you put 1 here you won't spot your mistake until you try this with conv2d!
    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum(
        "batch in_channels output_width kernel_width, out_channels in_channels kernel_width -> batch out_channels output_width", 
        x_strided, weights
    )

def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    
    batch, in_channels, height, width = x.shape
    out_channels, in_channels_2, kernel_height, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = width - kernel_width + 1
    output_height = height - kernel_height + 1
    
    xsB, xsIC, xsH, xsW = x.stride() # B for batch, IC for input channels, H for height, W for width
    wsOC, wsIC, wsH, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)
    x_new_stride = (xsB, xsIC, xsH, xsW, xsH, xsW)
    
    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum(
        "batch in_channels output_height output_width kernel_height kernel_width, \
out_channels in_channels kernel_height kernel_width \
-> batch out_channels output_height output_width",
        x_strided, weights
    )

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    B, C, W = x.shape
    output = x.new_full(size=(B, C, left + W + right), fill_value=pad_value)
    output[..., left : left + W] = x
    # Note - you can't use `left:-right`, because `right` could be zero.
    return output
    
def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    B, C, H, W = x.shape
    output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
    output[..., top : top + H, left : left + W] = x
    return output

def conv1d(x: t.Tensor, weights: t.Tensor, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    
    x_padded = pad1d(x, left=padding, right=padding, pad_value=0)
    
    batch, in_channels, width = x_padded.shape
    out_channels, in_channels_2, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = 1 + (width - kernel_width) // stride
    # note, we assume padding is zero in the formula here, because we're working with input which has already been padded
    
    xsB, xsI, xsWi = x_padded.stride()
    wsO, wsI, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_width, kernel_width)
    x_new_stride = (xsB, xsI, xsWi * stride, xsWi)
    # Explanation for line above:
    #     we need to multiply the stride corresponding to the `output_width` dimension
    #     because this is the dimension that we're sliding the kernel along
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum("B IC OW wW, OC IC wW -> B OC OW", x_strided, weights)

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


def conv2d(x: t.Tensor, weights: t.Tensor, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''

    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    
    x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=0)
    
    batch, in_channels, height, width = x_padded.shape
    out_channels, in_channels_2, kernel_height, kernel_width = weights.shape
    assert in_channels == in_channels_2, "in_channels for x and weights don't match up"
    output_width = 1 + (width - kernel_width) // stride_w
    output_height = 1 + (height - kernel_height) // stride_h
    
    xsB, xsIC, xsH, xsW = x_padded.stride() # B for batch, IC for input channels, H for height, W for width
    wsOC, wsIC, wsH, wsW = weights.stride()
    
    x_new_shape = (batch, in_channels, output_height, output_width, kernel_height, kernel_width)
    x_new_stride = (xsB, xsIC, xsH * stride_h, xsW * stride_w, xsH, xsW)
    
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    return einsum("B IC OH OW wH wW, OC IC wH wW -> B OC OH OW", x_strided, weights)


def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    '''Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''

    if stride is None:
        stride = kernel_size
    stride_height, stride_width = force_pair(stride)
    padding_height, padding_width = force_pair(padding)
    kernel_height, kernel_width = force_pair(kernel_size)
    
    x_padded = pad2d(x, left=padding_width, right=padding_width, top=padding_height, bottom=padding_height, pad_value=-t.inf)
    
    batch, channels, height, width = x_padded.shape
    output_width = 1 + (width - kernel_width) // stride_width
    output_height = 1 + (height - kernel_height) // stride_height
    
    xsB, xsC, xsH, xsW = x_padded.stride()
    
    x_new_shape = (batch, channels, output_height, output_width, kernel_height, kernel_width)
    x_new_stride = (xsB, xsC, xsH * stride_height, xsW * stride_width, xsH, xsW)
    
    x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)
    
    output = t.amax(x_strided, dim=(-1, -2))
    return output


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)
    tests.test_conv2d_minimal(conv2d_minimal)
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)
    tests.test_conv1d(conv1d)
    tests.test_conv2d(conv2d)
    tests.test_maxpool2d(maxpool2d)







# %% SECTION 4: MAKING YOUR OWN MODULES

from torch import nn

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


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        shape = input.shape
        
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        
        shape_left = shape[:start_dim]
        shape_middle = functools.reduce(lambda x, y: x*y, shape[start_dim : end_dim+1])
        shape_right = shape[end_dim+1:]
        
        new_shape = shape_left + (shape_middle,) + shape_right
        
        return t.reshape(input, new_shape)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

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
        x = einsum("... in_features, out_features in_features -> ... out_features", x, self.weight)
        if self.bias is not None: x += self.bias
        return x

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


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
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")
    tests.test_relu(ReLU)
    tests.test_flatten(Flatten)
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)
    tests.test_conv2d_module(Conv2d)

# %%

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(in_features=16*14*14, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.fc(self.flatten(self.relu(self.maxpool(self.conv(x)))))

if MAIN:
    model = SimpleCNN()
    print(model)

# %%

def get_mnist():
    '''Returns MNIST training data'''
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)
    return mnist_trainset, mnist_testset

if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    img, label = mnist_trainset[1]
    px.imshow(img.squeeze(), color_continuous_scale="gray", title=f"Label = {label}", width=500).show()

# %%

def write_to_html(fig, filename):
    with open(f"{filename}.html", "w") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
if MAIN:
    logits = model(img.unsqueeze(0)).squeeze().detach()
    probs = logits.softmax(-1)

    px.bar(
        y=probs, x=range(1, 11), height=400, width=600, template="ggplot2",
        title="Classification probabilities", labels={"x": "Digit", "y": "Probability"}, text_auto='.2f'
    ).update_layout(
        showlegend=False, xaxis_tickmode="linear"
    ).show()

# %%

if MAIN:
    BATCH_SIZE = 64
    NUM_SAMPLES = 2000
    NUM_BATCHES = NUM_SAMPLES // BATCH_SIZE

    mnist_trainloader = DataLoader(mnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []

    for i, (imgs, labels) in zip(range(NUM_BATCHES), mnist_trainloader):
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        print(f"Batches seen = {i+1:02}/{NUM_BATCHES}, Loss = {loss:.3f}")

    px.line(
        y=loss_list, x=range(BATCH_SIZE, NUM_SAMPLES, BATCH_SIZE),
        labels={"y": "Cross entropy loss", "x": "Num images seen"}, title="MNIST training curve (cross entropy loss)", template="ggplot2"
    ).update_layout(
        showlegend=False, yaxis_range=[0, max(loss_list)*1.1], height=400, width=600, xaxis_range=[0, NUM_SAMPLES]
    ).show()

# %%

if MAIN:
    logits = model(img.unsqueeze(0)).squeeze().detach()
    probs = logits.softmax(-1)

    px.bar(
        y=probs, x=range(1, 11), height=400, width=600, template="ggplot2",
        title="Classification probabilities", labels={"x": "Digit", "y": "Probability"},
    ).update_layout(
        showlegend=False, xaxis_tickmode="linear"
    ).show()

# %%