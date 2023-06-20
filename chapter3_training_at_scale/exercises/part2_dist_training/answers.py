#%%
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
# %% THREADS
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
# %% BROADCAST
from test import test_broadcast_naive

def broadcast_naive(tensor: torch.Tensor, src: int):
    current_rank = dist.get_rank()
    if src == current_rank:
        # Send to all other ranks
        for rank in range(dist.get_world_size()):
            if rank != src:
                dist.send(tensor, rank)
    else:
        # Receive tensor
        dist.recv(tensor, src)

if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
from test import test_broadcast_tree

def broadcast_tree(tensor: torch.Tensor, src: int):
    # SOLUTION
    curr_mult = 1
    rank_shifted = lambda: (dist.get_rank() - src) % dist.get_world_size()
    while curr_mult * 2 <= dist.get_world_size():
        if rank_shifted() < curr_mult:
            dist.send(tensor, (dist.get_rank() + curr_mult) % dist.get_world_size())
        elif rank_shifted() < curr_mult * 2:
            dist.recv(tensor, (dist.get_rank() - curr_mult) % dist.get_world_size())
        curr_mult *= 2
        dist.barrier()
# def broadcast_tree(tensor: torch.Tensor, src: int):
#     current_rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     print(f"{world_size=}")
#     print(f"{src=}")
#     if current_rank != src:
#         receive_from = current_rank // 2 if current_rank % 2 == 1 else (current_rank // 2) - 1
#         receive_from = receive_from % world_size
#         print(f"Receiving from {receive_from} to {current_rank}")
#         dist.recv(tensor, receive_from)

#     node1 = ((current_rank * 2) + 1) % world_size
#     node2 = ((current_rank * 2) + 2) % world_size

#     if current_rank >= src or node1 < src:
#         print(f"Sending from {current_rank} to {node1}")
#         dist.send(tensor, node1)

#     if current_rank >= src or node2 < src:
#         print(f"Sending from {current_rank} to {node2}")
#         dist.send(tensor, node2)
#     # 1 -> 2, 3; 2 -> 4, 3 -> 5, 
#     # j = 1
#     # for source_node in range(world_size):
#     #     for _ in range(2):
#     #         if source_node == current_rank:
#     #             print(f"{current_rank} Sending to {source_node + j}")
#     #             dist.send(tensor, source_node + j)
#     #             break
#     #         if current_rank == source_node + j:
#     #             print(f'{current_rank} Receiving from {source_node}')
#     #             dist.recv(tensor, source_node)
#     #         j += 1


if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    current_rank = dist.get_rank()
    world_size = dist.get_world_size()

    send_to = (current_rank + 1) % world_size
    receive_from = (current_rank - 1) % world_size 

    if send_to != src:
        dist.send(tensor, send_to)

    if current_rank != src:
        dist.recv(tensor, receive_from)

if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)
# %% REDUCTION
from test import test_reduce_naive

def do_operation(receive_tensor: torch.Tensor, tensor: torch.Tensor, op: ReduceOp):
    if op == ReduceOp.SUM:
        tensor += receive_tensor
    elif op == ReduceOp.PRODUCT:
        tensor *= receive_tensor
    elif op == ReduceOp.MAX:
        torch.max(receive_tensor, tensor, out=tensor)
    elif op == ReduceOp.MIN:
        torch.min(receive_tensor, tensor, out=tensor)
        
def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    current_rank = dist.get_rank()
    if dst == current_rank:
        # Send to all other ranks
        for rank in range(dist.get_world_size()):
            receive_tensor = torch.zeros_like(tensor)
            if rank != dst:
                dist.recv(receive_tensor, rank)
                do_operation(receive_tensor, tensor, op)
    else:
        # Receive tensor
        dist.send(tensor, dst)


if __name__ == '__main__':
    test_reduce_naive(reduce_naive)
# %%
from test import test_reduce_tree

def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):

    curr_mult = dist.get_world_size() // 2
    rank_shifted = (dist.get_rank() - dst) % dist.get_world_size()
    current_rank = dist.get_rank()
    world_size = dist.get_world_size()

    while curr_mult >= 1:
        if rank_shifted < curr_mult:
            receive_tensor = torch.zeros_like(tensor)
            dist.recv(receive_tensor, (current_rank + curr_mult) % world_size)
            do_operation(receive_tensor, tensor, op)
        elif rank_shifted < curr_mult * 2:
            dist.send(tensor, (current_rank - curr_mult) % world_size)
        curr_mult /= 2
        dist.barrier()
if __name__ == '__main__':
    test_reduce_tree(reduce_tree)
# %%
from test import test_allreduce_naive

def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    reduce_naive(tensor, 0, op)
    broadcast_naive(tensor, 0)

if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)
# %%
from test import test_allreduce_butterfly

def subset_naive_reduce(tensor: torch.Tensor, node1: int, node2: int, op: ReduceOp):
    current_rank = dist.get_rank()

    if current_rank == node2:
        # Send to all other ranks
        receive_tensor = torch.zeros_like(tensor)
        dist.recv(receive_tensor, node1)
        do_operation(receive_tensor, tensor, op)
    elif current_rank == node1:
        dist.send(tensor, node2)

# def subset_naive_broadcast(tensor: torch.Tensor, source: int, dst: int):
#     current_rank = dist.get_rank()

#     if current_rank == source:
#         dist.send(tensor, dst)
#     else:
#         dist.receive(tensor, source)

import numpy as np
def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):
    curr_shift = 1
    world_size = dist.get_world_size()

    n = int(np.log2(world_size))
    for _ in range(n):
        pairs = []
        added_sources = set()
        for node1 in range(world_size):
            if not node1 in added_sources:
                node2 = (node1 + curr_shift) % world_size
                pairs.append((node1, node2))
                added_sources.add(node1)
                added_sources.add(node2)

        for node1, node2 in pairs:
            subset_naive_reduce(tensor, node1, node2, op)
            subset_naive_reduce(tensor, node2, node1, op)

        curr_shift *= 2

if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)
# %%
