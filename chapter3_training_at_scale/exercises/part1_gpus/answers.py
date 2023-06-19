#%%
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
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# %%
inputs = torch.randn(5, 3, 224, 224)
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        with torch.inference_mode():
            model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
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
# %%
model = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

output = prof.key_averages().table(sOrt_by="self_cuda_time_total", row_limit=10)
print(output)
prof.export_chrome_trace("trace.json")
# %%

#### KERNEL FUSION AND BENCHMARKING

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