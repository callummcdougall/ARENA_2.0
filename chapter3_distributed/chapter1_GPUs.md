# GPUs

## Setup

```
import torch
import torchvision
from torch.utils import benchmark

from torch.profiler import profile, record_function, ProfilerActivity
import os

os.chdir('/root/dist/ARENA_2.0/')
from chapter0_fundamentals.exercises.part3_resnets.solutions import ResNet34
os.chdir('/root/dist/ARENA_2.0/chapter3_distributed')

```

# Section 1: Profiling - ATen out of Ten

Learning objectives:

1. Learn how to profile PyTorch model and view traces 
2. Understand the flow of control between the CPU and GPU
3. Understand the implementation levels in PyTorch

Readings:

1. http://blog.ezyang.com/2019/05/pytorch-internals/ (Required: Section - Finding your way around)
2. https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

## Looking at a pretrained torchvision ResNet model

### Profiling CPU time

Copy the below code to profile a pretrained ResNet-18 model from torchvision. The first section profiles the model and returns the 10 kernels that take up the most execution time. Remember some of these functions might be calling other functions and it's not completely obvious which ones these are.

```
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

We now apply an optimisation to the code i.e. running it in inference mode. Think about how this would change the execution time of different kernels in the model and where you would expect to see the results of this optimisation.
Think about what running a model in inference mode does and the answer should present itself to you. Run the code below to compare your expectations with reality.

```
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

```
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# %%
inputs = torch.randn(5, 3, 224, 224).cuda()
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
      with torch.inference_mode():
        model(inputs)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

Let's look at another way to profile and understand our model, the profile trace. Run the code below to generate a json file containing our trace. You can open this JSON file on any compatible web browser (Chrome or Edge) by going to the chrome://tracing webpage. The webpage should have a 'load JSON file' field that you can use to open the JSON file and view the profile trace. The profile trace is an extremely rich description of the model execution and helps us understand which functions are calling which other functions. 

```
# %%
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

```
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace_afterwarmup.json")
```

### Exercise: Profile your ResNet-34 model from Chapter 0

Difficulty: 3/5
Importance: 4/5

You should spend up to 15-20 minutes on this exercise.

Currently we have imported the ResNet-34 file from solutions.py in chapter0_fundamentals folder. If you have a version of ResNet-34 that you've made yourself, please change the import line:

```
from chapter0_fundamentals.exercises.part3_resnets.solutions import ResNet34
```
to
```
from chapter0_fundamentals.exercises.part3_resnets.<your file name> import ResNet34
```

```
# SOLUTION

# %%

model = ResNet34()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# %%

# %%
model = ResNet34().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")

output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
print(output)

```


# Section 2 - Kernel Fusion and Benchmarking

Learning objectives:

1. Learn how to perform PyTorch JIT kernel fusion
2. Understand the fundamentals of benchmarking and using torch.utils.benchmark
3. Create your own custom CUDA kernels

Readings:

1. https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-pointwise-operations
2. https://pytorch.org/tutorials/recipes/recipes/benchmark.html

## Demonstration on the daxpy function:
```
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

from torch.utils import benchmark


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

## Exercise: Apply kernel fusion by writing a custom kernel for naive softmax 

Difficulty: 4/5
Importance: 4/5

You should spend up to 15-20 minutes on this exercise.

```
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
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

# SOLUTION

```
@torch.jit.script
def fused_naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
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

Benchmarking scaffold:

```
def test_softmax_random_input(fn1, fn2):

  x = torch.randn(1823, 1823, device='cuda')

  assert torch.allclose(fn1(x), fn2(x), 0, 1e-6), "Implementations are not analogous"
  print('Tests passed')

from torch.utils import benchmark


test_softmax_random_input(naive_softmax, fused_naive_softmax)

print("benching...")
bench_results = []
for contender in [naive_softmax, fused_naive_softmax]:
  try:
    name = contender.__name__
  except:
    name = contender.name

  t = benchmark.Timer(
        setup="x = torch.randn(1823, 1823, device='cuda')",
        stmt="function(x)",
        description=f"cuda",
        label="naive_softmax",
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

# Section 3 - Quantization and AMP

Learning objective:

1. Understand the effects of quantisation on inference time
2. Using PyTorch AMP and understanding tradeoffs

Readings:
1. https://arxiv.org/pdf/1712.05877.pdf
2. 

## Exercise: Quantization from first-principles

To quantize and dequantize a tensor we use the following formula:

```
x_Float = scale*(x_Quant -zero_point). Hence,

x_Quant = (x_Float/scale) + zero_point.

Here scale is equal (max_val — min_val) / (qmax — qmin)
```

Where max_val and min_val are maximum and minimum values of X tensor respectively. qmin and q_max represents the range of an 8 bit number (0 and 255 respectively). The scale scales the quantised net and the zero point shifts the number. The dequantisation and quantisation functions given below give more clarity as how a floating point tensor is converted to an 8 bit tensor and vice versa.

Setup:

```
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
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
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
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
    
    return model
```

```
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def quantize_tensor(x, num_bits=8) -> QTensor:

    # SOLUTION
    qmin = 0.
    qmax = 2.**num_bits - 1.
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


def dequantize_tensor(q_x):
    # SOLUTION
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)
```

```
def calcScaleZeroPoint(min_val, max_val,num_bits=8):
  # Calc Scale and zero point 
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
  
def quantizeLayer(x, layer, stat, scale_x, zp_x):
  # should work for both conv and linear layers

  # SOLUTION

  W = layer.weight.data
  B = layer.bias.data

  # scale_x = x.scale
  # zp_x = x.zero_point
  w = quantize_tensor(layer.weight.data) 
  b = quantize_tensor(layer.bias.data)

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  scale_w = w.scale
  zp_w = w.zero_point
  
  scale_b = b.scale
  zp_b = b.zero_point
  

  scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

  # Perparing input by shifting
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


def quantForward(model, x, stats):
  
  # Quantise before inputting into incoming layers

  # SOLUTION

  x = quantize_tensor_act(x, stats['conv1'])

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

```
def testQuant(model, test_loader, quant=False, stats=None):
    device = 'cuda'
    
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
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

