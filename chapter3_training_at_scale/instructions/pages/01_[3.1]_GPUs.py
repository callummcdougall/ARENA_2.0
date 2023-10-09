
import os, sys
from pathlib import Path
chapter = r"chapter3_training_at_scale"
instructions_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/instructions").resolve()
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st
import st_dependencies

st_dependencies.styling()

import platform
is_local = (platform.processor() != "")

def section_0():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/gpu.png" width="350">


Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [3.1] - GPUs


## Introduction


This chapter contains exercises to get you familiar with the internal workings of GPUs and how they translate into PyTorch. We will profile PyTorch code and see how operation calls and training times change as we optimise certain parts of our code. We will also look into kernel fusion and benchmarking the optimisations you'll make. Lastly, we'll quantise a model to INT8 for a light, fast model that can be deployed to the CPU and operate very efficiently. The bonus material aims to give you more exposure into other optimisation tricks and estimating compute for those inclined towards forecasting.


## Content & Learning Objectives


#### 1️⃣ Profiling - ATen out of Ten

> ##### Learning objectives
> 
> - Learn how to profile PyTorch model and view traces 
> - Understand the flow of control between the CPU and GPU
> - Understand the implementation levels in PyTorch

#### 2️⃣ Kernel Fusion and Benchmarking

> ##### Learning objectives
> 
> - Learn how to perform PyTorch JIT kernel fusion
> - Understand the fundamentals of benchmarking and using torch.utils.benchmark
> - Create your own custom CUDA kernels

#### 3️⃣ Quantization

> ##### Learning objectives
> 
> - Understand the effects of quantisation on inference time

#### 4️⃣ Bonus

Includes some suggested bonus exercises and further reading.


## Setup


```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch
import torchvision
from torch.utils import benchmark
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import copy

from collections import namedtuple

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path

from typing import List, Optional, Callable, Tuple, Dict, Literal, Set 
# Make sure exercises are in the path
orig_dir = os.getcwd()
chapter = r"chapter3_training_at_scale"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part1_gpus.tests as tests

# Add root dir, so we can import from chapter 0 material
root_dir = exercises_dir.parent.parent.resolve()
if str(root_dir) not in sys.path: sys.path.append(str(root_dir))
os.chdir(root_dir)
from chapter0_fundamentals.exercises.part3_resnets.solutions import ResNet34
os.chdir(orig_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

```



""", unsafe_allow_html=True)


def section_1():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#reading'>Reading</a></li>
    <li class='margtop'><a class='contents-el' href='#looking-at-a-pretrained-torchvision-resnet-model'>Looking at a pretrained torchvision ResNet model</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#profiling-cpu-time'>Profiling CPU time</a></li>
        <li><a class='contents-el' href='#profiling-cpu-and-gpu-time'>Profiling CPU and GPU time</a></li>
        <li><a class='contents-el' href='#exercise:-profile-your-resnet-12510-model-from-chapter-0'><b>Exercise</b>: Profile your ResNet-34 model from Chapter 0</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 1️⃣ Profiling - ATen out of Ten


> ##### Learning objectives
> 
> - Learn how to profile PyTorch model and view traces 
> - Understand the flow of control between the CPU and GPU
> - Understand the implementation levels in PyTorch


## Reading

1. [ezyang's blog - PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/) (Required: Section - Finding your way around)
2. [PyTorch docs - Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)


## Looking at a pretrained torchvision ResNet model


### Profiling CPU time

Copy the below code to profile a pretrained ResNet-18 model from torchvision. The first section profiles the model and returns the 10 kernels that take up the most execution time. Remember some of these functions might be calling other functions and it's not completely obvious which ones these are.


```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

We now apply an optimisation to the code i.e. running it in inference mode. Think about how this would change the execution time of different kernels in the model and where you would expect to see the results of this optimisation.

Think about what running a model in inference mode does and the answer should present itself to you. Run the code below to compare your expectations with reality.


```python
inputs = torch.randn(5, 3, 224, 224)
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        with torch.inference_mode():
            model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### Profiling CPU and GPU time

Let's see what happens when we profile both CPU and CUDA operations and add in our optimisation. The tables that you will see now will have more columns than before and show you separate CPU and GPU times for certain functions. 


```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

```python
inputs = torch.randn(5, 3, 224, 224).cuda()
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        with torch.inference_mode():
            model(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

Let's look at another way to profile and understand our model, the profile trace. Run the code below to generate a json file containing our trace. You can open this JSON file on any compatible web browser (Chrome or Edge) by going to the `chrome://tracing` webpage. The webpage should have a 'load JSON file' field that you can use to open the JSON file and view the profile trace. The profile trace is an extremely rich description of the model execution and helps us understand which functions are calling which other functions. 


```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
print(output)
prof.export_chrome_trace("trace.json")
```

You might have noticed an extremely large section of non-activity in the beginning of the trace and this is due to warmup. The first time you run a model on a GPU various kernels are compiled in the way that the model needs them to function, this compilation time adds a large overhead over the first execution of the model on the GPU. Get another trace from the profiler and this overhead should be removed now.

NOTE: Depending on the order of execution of these cells, you might not notice any warmup but rest assured warmup is a guaranteed overhead over all the first runs of a model on a GPU.


```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace_afterwarmup.json")
```

### Exercise: Profile your ResNet-34 model from Chapter 0

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 15-20 minutes on this exercise.
```

Currently we have imported the ResNet-34 file from solutions.py in chapter0_fundamentals folder. If you have a version of ResNet-34 that you've made yourself, please change the import line:

```python
from chapter0_fundamentals.exercises.part3_resnets.solutions import ResNet34
```

to

```python
from chapter0_fundamentals.exercises.part3_resnets.<your-file-name> import ResNet34
```

It is imperative that you identify bottlenecks and post screenshots of your traces in the #memes channel


```python
model = ResNet34()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

```python
model = ResNet34().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")

output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
print(output)
```



""", unsafe_allow_html=True)


def section_2():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#readings'>Readings</a></li>
    <li class='margtop'><a class='contents-el' href='#demonstration-on-the-daxpy-function:'>Demonstration on the daxpy function:</a></li>
    <li class='margtop'><a class='contents-el' href='#exercise:-apply-kernel-fusion-by-using-torch-jit-decorator'><b>Exercise</b>: Apply kernel fusion by using torch jit decorator</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 2️⃣ Kernel Fusion and Benchmarking


> ##### Learning objectives
> 
> - Learn how to perform PyTorch JIT kernel fusion
> - Understand the fundamentals of benchmarking and using torch.utils.benchmark
> - Create your own custom CUDA kernels


## Readings

1. [PyTorch - Perfomance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-pointwise-operations)
2. [PyTorch docs - Benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)


## Demonstration on the daxpy function:

Look into the test_daxpy_random_input and try experimenting with the arguments of the torch.allclose function to see the differences in the output of the fused and non-fused functions.


```python
def daxpy(alpha,x,y):
  return alpha*x + y

@torch.jit.script
def fused_daxpy(alpha,x,y):
    return alpha*x + y

def test_daxpy_random_input(fn1, fn2):

    alpha = torch.rand(1, device='cuda')
    x = torch.randn(1823, 1823, device='cuda')
    y = torch.randn(1823, 1823, device='cuda')

    assert torch.allclose(fn1(alpha, x, y), fn2(alpha, x, y), 0, 1e-6), "Implementations are not analogous"
    print('Tests passed')

test_daxpy_random_input(daxpy, fused_daxpy)

print("benching...")
bench_results = []
for contender in [daxpy, fused_daxpy]:
    try:
        name = contender.__name__
    except:
        name = contender.name

    t = benchmark.Timer(
        setup="alpha, x, y = torch.rand(1, device='cuda'),torch.randn(1823, 1823, device='cuda'), torch.randn(1823, 1823, device='cuda') ",
        stmt="function(alpha, x, y)",
        description=f"cuda",
        label="daxpy",
        sub_label=name,
        globals={
            'function': contender
        }
      )
    bench_results.append(t.blocked_autorange(min_run_time=5))


compare = benchmark.Compare(bench_results)
compare.colorize()
compare.print()
```

## Exercise: Apply kernel fusion by using torch jit decorator
```yaml
Difficulty: 2/5
Importance: 🔵🔵🔵🔵⚪

You should spend up to 15-20 minutes on this exercise.
```

Use decorators to perform kernel fusion compilers and compare them using the benchmark function used above.


```python
def naive_softmax(x):
    '''Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    '''
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

```

<details>
<summary>Solution </summary>

PyTorch JIT solution:

```python
@torch.jit.script
def torchjit_fused_naive_softmax(x):
    '''Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    '''
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
```

Benchmarking code:

```python

print("benching...")
bench_results = []
for contender in [naive_softmax, torchjit_fused_naive_softmax]:
  try:
    name = contender.__name__
  except:
    name = contender.name

  t = benchmark.Timer(
        setup="x = torch.randn(1823, 1823, device='cuda')",
        stmt="function(x)",
        description=f"cuda",
        label="softmax",
        sub_label=name,
        globals={
            'function': contender
        }
    )
  bench_results.append(t.blocked_autorange(min_run_time=5))


compare = benchmark.Compare(bench_results)
compare.colorize()
compare.print()
```

</details>




""", unsafe_allow_html=True)


def section_3():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#readings'>Readings</a></li>
    <li class='margtop'><a class='contents-el' href='#quantization-from-first-principles'>Quantization from first-principles</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-fill-in-quantize-and-dequantize-functions'><b>Exercise</b> - fill in <code>quantize</code> and <code>dequantize</code> functions</a></li>
        <li><a class='contents-el' href='#exercise-finish-the-calcscalezeropoint-function'><b>Exercise</b> - Finish the calcScaleZeroPoint function</a></li>
        <li><a class='contents-el' href='#provided-the-gatherstats-and-gatheractivations-functions'>Provided - The <code>gatherstats</code> and <code>gatherActivations</code> functions</a></li>
        <li><a class='contents-el' href='#provided-initialising-the-q-model-and-gathering-stats'>Provided - Initialising the q_model and gathering stats</a></li>
        <li><a class='contents-el' href='#exercise-fill-out-the-quantizelayer-function'><b>Exercise</b> - Fill out the quantizeLayer function</a></li>
        <li><a class='contents-el' href='#provided-the-quantforward-function'>Provided - The quantForward function</a></li>
        <li><a class='contents-el' href='#exercise-benchmark-q-model-and-model'><b>Exercise</b> - Benchmark q_model and model</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 3️⃣ Quantization and 


> ##### Learning objectives
> 
> - Understand the effects of quantisation on inference time


## Readings

1. [Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference
](https://arxiv.org/pdf/1712.05877.pdf)


## Quantization from first-principles


We will start with a small convolutional neural network trained on MNIST. We provide setup code to train the model and the following exercises will guide you through quantising your model to INT8..


Setup:


```python
class Net(nn.Module):
    def __init__(self, mnist=True):
      
        super(Net, self).__init__()
        if mnist:
            num_channels = 1
        else:
            num_channels = 3
          
        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

      
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)   
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))


def test(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

def main():
 
    batch_size = 64
    test_batch_size = 64
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = False
    no_cuda = False
    
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )),
        batch_size=test_batch_size,
        shuffle=True, 
        **kwargs
    )
    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
    
    return model


model = main()
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data', 
        train=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=64, shuffle=True)

test(model, test_loader)
```

### Exercise - fill in `quantize` and `dequantize` functions

```yaml
Difficulty: 🔴🔴🔴🔴🔴
Importance: 🔵🔵🔵🔵⚪

You should spend up to 15-20 minutes on this exercise.
```


To quantize and dequantize a tensor we use the following formula:

```
x_Float = scale*(x_Quant -zero_point)
```

hence, we have

```
x_Quant = (x_Float/scale) + zero_point
```

Here `scale` is equal to `(max_val — min_val) / (qmax — qmin)`, where `max_val` and `min_val` are maximum and minimum values of X tensor respectively. `qmin` and `qmax` represents the range of an 8 bit number (0 and 255 respectively). The scale scales the quantised net and the zero point shifts the number. The dequantisation and quantisation functions given below give more clarity as how a floating point tensor is converted to an 8 bit tensor and vice versa.

The QTensor namedtuple is a clean abstraction that lets you store scale and zero_point information along with the original tensor.


```python

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])
    
def quantize_tensor(x, min_val=None, max_val=None, num_bits=8) -> QTensor:
    '''
    Calculate the scale and zero-point of the input tensor for quantization.
    '''
    pass


tests.test_quantize_tensor(quantize_tensor)


def dequantize_tensor(q_x) -> torch.tensor:
    '''
    Dequantize the input QTensor to obtain the float Tensor.
    '''
    pass
    

tests.test_dequantize_tensor(dequantize_tensor)
```

<details>
<summary>Solution</summary>


```python
def quantize_tensor(x, min_val=None, max_val=None, num_bits=8) -> QTensor:
    '''
    Calculate the scale and zero-point of the input tensor for quantization.
    '''
    # SOLUTION
    qmin = 0.
    qmax = 2.**num_bits - 1.

    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()
    
    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

    
def dequantize_tensor(q_x) -> torch.tensor:
    '''
    Dequantize the input QTensor to obtain the float Tensor.
    '''
    # SOLUTION
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)
```
</details>


### Exercise - Finish the calcScaleZeroPoint function

We'll be using the `calcScaleZeroPoint` function to calculate the values of the scale and zero point for a given `max_val` and `min_val` of a tensor. The function should work for `num_bits=8` as a baseline but can be lowered for harsher quantisation.

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 15-20 minutes on this exercise.
```


```python
def calcScaleZeroPoint(min_val, max_val, num_bits=8) -> Tuple[float, float]:
    '''
    Calculate scale and zero point of
    '''
    pass


```

### Provided - The <code>gatherstats</code> and <code>gatherActivations</code> functions

We provide the function to gather stats regarding the minimum and maximum values of all the tensors that are used in the forward passs of our model. We will use these stats to quantize our model in the next exercise.


```python
# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key) -> Dict[Dict, Dict[str, int]]:
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)
    
    
    if key not in stats:
        stats[key] = {"max": max_val.sum().item(), "min": min_val.sum().item(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1
    
    return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = F.relu(model.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = F.relu(model.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4*4*50)
    stats = updateStats(x, stats, 'fc1')
    x = F.relu(model.fc1(x))
    stats = updateStats(x, stats, 'fc2')
    x = model.fc2(x)

    return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda'
    
    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)
    
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return final_stats

```

### Provided - Initialising the q_model and gathering stats

We initialise the model to be quantised as q_model and copy it directly from our trained model. We also use this new model to gather stats from all the layers in our model.


```python
q_model = copy.deepcopy(model)
stats = gatherStats(q_model, test_loader)
```

### Exercise - Fill out the quantizeLayer function

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 30-50 minutes on this exercise.
```

The quantizeLayer function will take in an input tensor, the layer it's being input into and quantize the input, weights and biases according to the stats we captured using the previous functions. After quantisation, we execute a forward pass to get the output tensor of the layer.

Steps to follow:

1. Shift the input tensor using the zp_x argument
2. Calculate the scale and zero point of the next layer using the calcScaleZeroPoint function
3. Quantize the weights and biases of the layer using the calculated scale and zero point
4. Execute a forward pass of the input tensor through the quantized layer, apply a ReLU and obtain the output tensor
5. Finally, we reset the layers weights and biases to their original values
6. Return the output tensor, scale and zero point of the next layer


```python
def quantizeLayer(x, layer, stat, scale_x, zp_x) -> Tuple[torch.tensor, float, float]:
    '''
    Should work for both conv and linear layers.
    '''
    pass


```

<details>
<summary>Solution</summary>


```python
def calcScaleZeroPoint(min_val, max_val, num_bits=8) -> Tuple[float, float]:
    '''
    Calculate scale and zero point of
    '''
    # SOLUTION

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale_next = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale_next

    zero_point_next = 0
    if initial_zero_point < qmin:
        zero_point_next = qmin
    elif initial_zero_point > qmax:
        zero_point_next = qmax
    else:
        zero_point_next = initial_zero_point

    zero_point_next = int(zero_point_next)

    return scale_next, zero_point_next

def quantizeLayer(x, layer, stat, scale_x, zp_x) -> Tuple[torch.tensor, float, float]:
    '''
    Should work for both conv and linear layers.
    '''
    # SOLUTION

    W = layer.weight.data
    B = layer.bias.data
    
    w = quantize_tensor(layer.weight.data) 
    b = quantize_tensor(layer.bias.data)

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    scale_w = w.scale
    zp_w = w.zero_point

    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

    # Preparing input by shifting by zero point
    X = x.float() - zp_x

    
    layer.weight.data = (scale_x * scale_w/scale_next)*(layer.weight.data - zp_w)
    layer.bias.data = (scale_b/scale_next)*(layer.bias.data + zp_b)

    # All int

    x = layer(X) + zero_point_next
    x = F.relu(x)

    # Reset
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next
```
</details>


### Provided - The quantForward function

All the functions you implemented before come together in this function, use the details of this function to ensure that all your other functions return the correct things. Notice that the stats passed into the quantizeLayer function belong to the next layer because we first quantize the output of the previous layer then 

```python
def quantForward(model, x, stats):
    '''
    Quantise before inputting into incoming layers
    '''
    x = quantize_tensor(x, min_val = stats['conv1']['min'], max_val = stats['conv1']['max'])

    x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv2'], x.scale, x.zero_point)

    x = F.max_pool2d(x, 2, 2)
    
    x, scale_next, zero_point_next = quantizeLayer(x, model.conv2, stats['fc1'], scale_next, zero_point_next)

    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 4*4*50)

    x, scale_next, zero_point_next = quantizeLayer(x, model.fc1, stats['fc2'], scale_next, zero_point_next)
    
    # Back to dequant for final layer
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
    
    x = model.fc2(x)

    return F.log_softmax(x, dim=1)
```


Test your model using the below code:


```python
def testQuant(model, test_loader, device='cuda', quant=False, stats=None):

    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
              output = quantForward(model, data, stats)
            else:
              output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) #bm get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
  

testQuant(model, test_loader=test_loader, quant=True, stats=stats)
```

### Exercise - Benchmark q_model and model

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 10-15 minutes on this exercise.
```

We will now benchmark our quanitized model and our vanilla model. It's important to toy around with `device` being sent as an argument to the test functions and `num_threads` to understand in which scenarios our quantized model performs significantly better than our original model.

<details>
<summary>Hint</summary>

HINT: Quantization helps in resource constrained scenarios

</details>


```python
num_threads = 10 # Change to see different benchmarking results
print(f'Benchmarking on {num_threads} threads')

t0 = benchmark.Timer(
    stmt='test(model, test_loader)',
    setup='from __main__ import test, model, test_loader',
    num_threads=num_threads,
    label='Vanilla model')

t1 = benchmark.Timer(
    stmt='testQuant(q_model, test_loader, quant=True, stats=stats)',
    setup='from __main__ import testQuant, q_model, test_loader, stats',
    num_threads=num_threads,
    label='INT8 Quantized model')


print(t0.timeit(5))
print(t1.timeit(5))
```

<details>
<summary>Solution</summary>

```python
# SOLUTION
num_threads = 1 # Change to see different benchmarking results
print(f'Benchmarking on {num_threads} threads')

t0 = benchmark.Timer(
    stmt='test(model, test_loader, device=\'cpu\')',
    setup='from __main__ import test, model, test_loader',
    num_threads=num_threads,
    label='Vanilla model')

t1 = benchmark.Timer(
    stmt='testQuant(q_model, test_loader, quant=True, stats=stats, device=\'cpu\')',
    setup='from __main__ import testQuant, q_model, test_loader, stats',
    num_threads=num_threads,
    label='INT8 Quantized model')


print(t0.timeit(5))
print(t1.timeit(5))
```
</details>




""", unsafe_allow_html=True)


def section_4():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#estimating-compute'>Estimating Compute</a></li>
    <li class='margtop'><a class='contents-el' href='#brrrr-ing-your-dataloader'>Brrrr-ing your dataloader</a></li>
    <li class='margtop'><a class='contents-el' href='#pytorch-amp-and-channels-last'>PyTorch AMP and Channels Last</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 4️⃣ Bonus section

## Create your own custom kernel for kernel fusion

Go through the CUDA workshop [material](https://colab.research.google.com/gist/pranavgade20/aac9c6532e8cb262dcb7adeda9b9edba/gpu_basics.ipynb) and use the techniques discussed here to create your own kernel.

Another way to perform kernel fusion is usung Torchscript, here's a great article listing the [basics](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).

## Estimating Compute

Readings:

1. [Eleuther AI - Transformer Math](https://blog.eleuther.ai/transformer-math/)
2. [Kipp,ly - Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)
3. [Estimating training compute of Deep Learning models](https://docs.google.com/document/d/1J2BX9jkE5nN5EA1zYRN0lHhdCf1YkiFERc_nwiYqCOA/edit)

Estimating compute for large training runs is a large part of forecasting literature and the readings above contain almost all of the high-quality readings on this topic.

Suggested exercises:

1. Find a model not on [Epoch AI's datasheet](https://docs.google.com/spreadsheets/d/1AAIebjNsnJj_uKALHbXNfn3_YsT6sHXtCU0q7OIPuc4/edit#gid=0) and contribute an entry to it
2. Verify the theoretical estimate of a model's compute by training your own and calculating the FLOPs

## Brrrr-ing your dataloader

1. [Multiprocessing and Memory pinning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)

Use multiprocessing and memory pinning to make your dataloaders fast!

## PyTorch AMP and Channels Last

1. [Channels last format](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-channels-last-memory-format-for-computer-vision-models)
2. [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)

Automatic Mixed Precision is an easy to use tool that automatically quantizes models during training time to cut down training time by a large margin (upto ~4x). Try using PyTorch AMP to train the ResNet model from Week 0 faster.




""", unsafe_allow_html=True)


func_page_list = [
    (section_0, "🏠 Home"),     (section_1, "1️⃣ Profiling - ATen out of Ten"),     (section_2, "2️⃣ Kernel Fusion and Benchmarking"),     (section_3, "3️⃣ Quantization"),     (section_4, "4️⃣ Bonus section"), 
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = dict(zip(page_list, range(len(page_list))))

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    idx = page_dict[radio]
    func = func_list[idx]
    func()

page()