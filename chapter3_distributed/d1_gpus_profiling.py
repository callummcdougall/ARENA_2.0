# %%

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os

os.chdir('/root/dist/ARENA_2.0/')
from chapter0_fundamentals.exercises.part3_resnets.solutions import ResNet34
os.chdir('/root/dist/ARENA_2.0/chapter3_distributed')

# %%

model = ResNet34()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)
# %%
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# %%
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

# %%
model = ResNet34().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")
# %%
model = ResNet34().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

prof.export_chrome_trace("trace_afterwarmup.json")

# %%

output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
print(output)
# %%
from torch.profiler import schedule

my_schedule = schedule(
    skip_first=10,
    wait=5,
    warmup=1,
    active=3,
    repeat=2)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=trace_handler
) as p:
    for idx in range(8):
        model(inputs)
        p.step()

# %%
