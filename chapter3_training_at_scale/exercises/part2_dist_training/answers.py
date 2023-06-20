# %%
import sys
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp
from typing import List
from pathlib import Path
import gdown

gdown.download("https://drive.google.com/file/d/1QgkqHSPDwQD-Z0K0-4CUhp8fW-X0hWds/view", '/tmp/libnccl.so.2.18.1', quiet=False, fuzzy=True)
gdown.download("https://drive.google.com/file/d/1tqUv0OktQdarW8hUyHjqNnxDP1JyUdkq/view?usp=sharing", quiet=False, fuzzy=True)

# Make sure exercises are in the path
chapter = r"chapter3_training_at_scale"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
from threading import Thread

# Add to the global variable
def adder(amount, repeats):
    global value
    for _ in range(repeats):
        value += amount

# Subtract from the global variable
def subtractor(amount, repeats):
    global value
    for _ in range(repeats):
        value -= amount

def add_and_subtract():
    # Start a thread making additions
    adder_thread = Thread(target=adder, args=(1, 1000000))
    adder_thread.start()
    # Start a thread making subtractions
    subtractor_thread = Thread(target=subtractor, args=(1, 1000000))
    subtractor_thread.start()
    # Wait for both threads to finish
    print('Waiting for threads to finish...')
    adder_thread.join()
    subtractor_thread.join()
    # Print the value
    print(f'Value: {value}')


if __name__ == '__main__':
    value = 0
    add_and_subtract()
# %%

from test import test_broadcast_naive

def broadcast_naive(tensor: torch.Tensor, src: int):
    me = dist.get_rank()
    if src != me:
        dist.recv(tensor, src)
    else:
        for i in range(dist.get_world_size()):
            if src == me:
                continue
            dist.send(tensor, i)
    


if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)

# %%

from test import test_broadcast_tree
import math
def broadcast_tree(tensor: torch.Tensor, src: int):
    me = dist.get_rank()
    m = dist.get_world_size()

    def call_child(c: int):
        if c >= m:
            return
        if d == src:
            return
        dist.send(tensor, d)
        

    my_caller = me // 2
    if my_caller == src:

    dist.recv(tensor, my_caller)
    for d in [me * 2, me * 2 + 1]:
        call_child(d)



if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
# %%

from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    me = dist.get_rank()    
    if me != src:
        dist.recv(tensor, (me - 1) % dist.get_world_size())
    d = (me + 1) % dist.get_world_size()
    if d != src:
        dist.send(tensor, d)
    dist.barrier


if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)


# %%
from test import test_reduce_naive

def reduction_op(op, tensor: torch.Tensor, other: torch.Tensor):
    if op == ReduceOp.SUM:
        tensor += other
    elif op == ReduceOp.PRODUCT:
        tensor *= other
    elif op == ReduceOp.MAX:
        torch.max(tensor, other, out=tensor)
    elif op == ReduceOp.MIN:
        torch.min(tensor, other, out=tensor)

def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    me = dist.get_rank()

    if me == dst:
        for i in range(dist.get_world_size()):
            if i == me:
                continue
            tensor_i = torch.zeros_like(tensor)
            dist.recv(tensor_i, i)
            reduction_op(op, tensor, tensor_i)
    else:
        dist.send(tensor, dst)


if __name__ == '__main__':
    test_reduce_naive(reduce_naive)
# %%
from test import test_allreduce_naive

def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    sync_rank = 0
    me = dist.get_rank()
    m = dist.get_world_size()

    if me == sync_rank:
        for i in range(m):
            if i == me:
                continue
            tensor_i = torch.zeros_like(tensor)
            dist.recv(tensor_i, i)
            reduction_op(op, tensor, tensor_i)
        for i in range(m):
            if i == me:
                continue
            dist.send(tensor, i)
    else:
        dist.send(tensor, sync_rank)
        dist.recv(tensor, sync_rank)


if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)
# %%
