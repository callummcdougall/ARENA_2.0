# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp
from typing import List
from pathlib import Path
import gdown
import sys

gdown.download("https://drive.google.com/file/d/1QgkqHSPDwQD-Z0K0-4CUhp8fW-X0hWds/view", '/tmp/libnccl.so.2.18.1', quiet=False, fuzzy=True)
# gdown.download("https://drive.google.com/file/d/1tqUv0OktQdarW8hUyHjqNnxDP1JyUdkq/view?usp=sharing", quiet=False, fuzzy=True)
# !wget http://192.9.158.9:8000/imagenet_38k.zip
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
    my_rank = dist.get_rank()
    if my_rank == src:
        for dst in range(dist.get_world_size()):
            if dst != my_rank:
                # print(f"sending to {dst}")
                dist.send(tensor, dst)
    else:
        dist.recv(tensor, src)
        


if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
# %%
from test import test_broadcast_tree

def broadcast_tree(tensor: torch.Tensor, src: int):
    """
    Broadcast using a binary tree.
    In step n, up to 2^n ranks have the tensor.
    """
    world_size = dist.get_world_size()
    # print(f"all processes = {dist.get_process_group_ranks(dist.distributed_c10d._get_default_group())}")
    my_rank = dist.get_rank()
    # assert my_rank != 0
    block_size = 1
    print(f"{dist.get_rank()}, {block_size=}")
    while block_size < world_size:
        partner = my_rank ^ block_size
        if partner < world_size:
            transmit = (my_rank | (block_size - 1)) == (src | (block_size - 1))
            if transmit:
                dist.send(tensor, partner)
            receive = (partner | (block_size - 1)) == (src | (block_size - 1))
            if receive:
                dist.recv(tensor, partner)
            print(f"{'sending' if transmit else 'recieving' if receive else ''} {my_rank=}, {partner=}, {block_size=}")
        block_size *= 2

if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
# %%
from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    world_size = dist.get_world_size()    
    my_rank = dist.get_rank()
    if my_rank != src:
        dist.recv(tensor, (my_rank - 1) % world_size)
    if my_rank != (src - 1) % world_size:
        dist.send(tensor, (my_rank + 1) % world_size)


if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)
# %%
from test import test_reduce_naive

def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    my_rank = dist.get_rank()
    world_size = dist.get_world_size()
    if my_rank != dst:
        dist.send(tensor, dst)
    else:
        buffer = torch.zeros_like(tensor)
        for src in range(world_size):
            if src != dst:
                dist.recv(buffer, src)
                if op == ReduceOp.SUM:
                    tensor += buffer
                else:
                    raise NotImplementedError
        
        


if __name__ == '__main__':
    test_reduce_naive(reduce_naive)
# %%

from test import test_reduce_tree

def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    world_size = dist.get_world_size()
    # print(f"all processes = {dist.get_process_group_ranks(dist.distributed_c10d._get_default_group())}")
    my_rank = dist.get_rank()
    # assert my_rank != 0
    block_size = 1
    print(f"{dist.get_rank()}, {block_size=}")
    while block_size < world_size:
        partner = my_rank ^ block_size
        block_size *= 2
        if partner < world_size:
            transmit = (partner & (block_size - 1)) == (dst & (block_size - 1))
            if transmit:
                dist.send(tensor, partner)
            receive = (my_rank & (block_size - 1)) == (dst & (block_size - 1))
            if receive:
                dist.recv(tensor, partner)
            print(f"{'sending' if transmit else 'recieving' if receive else ''} {my_rank=}, {partner=}, {block_size=}")



if __name__ == '__main__':
    test_reduce_tree(reduce_tree)
# %%
from test import test_allreduce_naive

def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):    
    reduce_naive(tensor, dst=0)
    dist.barrier()
    broadcast_naive(tensor, src=0)


if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)
# %%
from test import test_allreduce_butterfly

def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):
    """
    Performs all-reduce using a butterfly topology.
    In step i (0-indexed), each rank sums with its partner at distance block_size = 2**i.
    After step i, each block contains the sum of its block of size 2 * 2**i.
    """
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    block_size = 1
    while block_size < world_size:
        partner = my_rank ^ block_size
        buffer = torch.zeros_like(tensor)
        if partner < world_size:
            # 0 sends, 1 receives
            if my_rank & block_size == 0:
                dist.send(tensor.clone(), partner)
            else:
                dist.recv(buffer, partner)

            dist.barrier()

            # 1 sends, 0 receives
            if my_rank & block_size != 0:
                dist.send(tensor.clone(), partner)
                tensor += buffer
            else:
                dist.recv(buffer, partner)
                tensor += buffer
        block_size *= 2
        dist.barrier()
        

        
if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)
# %%
# import AutoModelForCausalLM
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
# %%
model
# %%
model.transformer.h[0]
# %%
