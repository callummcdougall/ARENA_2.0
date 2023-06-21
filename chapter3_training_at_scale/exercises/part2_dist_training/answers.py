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
### DATA PARALLELISM

import tqdm
import argparse
import os
import logging
import time
import random
import string

import torch.distributed as dist

import torch as t
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from torchvision.io import read_image

assert torch.cuda.device_count() > 0  # make sure we have GPUs

CLUSTER_SIZE = 1  # the number of separate compute nodes we have
WORLD_SIZE = 2  # the number of processes we want to launch - this is usually equal to the number of GPUs we have on this machine
TOTAL_RANKS = CLUSTER_SIZE * WORLD_SIZE
UNIGPU = torch.cuda.device_count() == 1  # remember to use the patched NCCL binary if you are using colab/practicing on a single GPU. You might need to compile https://github.com/pranavgade20/nccl-unigpu if you aren't using colab

def main(args):
    rank = args.rank

    world_size = args.world_size
    logging.basicConfig(format=f'[rank {args.rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    logging.warning(f'hello')
    # you can use a variety of backends - nccl is the fastest but requires you to have a GPU; gloo is slower but also sometimes works on CPUs (see https://pytorch.org/docs/stable/distributed.html#backends)
    # dist.init_process_group(backend='nccl', init_method=f'file:///tmp/{"".join(random.choice(string.ascii_letters) for _ in range(10))}', world_size=WORLD_SIZE, rank=args.rank)
    dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:12346', world_size=WORLD_SIZE, rank=args.rank)  # this should be a globally accessible IP
    logging.warning(f'distributed.is_initialized {torch.distributed.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {torch.distributed.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {torch.distributed.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {torch.distributed.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {torch.distributed.is_torchelastic_launched()}')

    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    file_mappings = json.load(open('/home/ubuntu/file_mappings_imagenet.json'))
    logging.warning("Loading Data:")

    imagenet_valset = list((lambda k=k: read_image(f'/home/ubuntu/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
    imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
    imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
    imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    logging.warning("Transforming Data:")
    imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]

    time.sleep(1)

    # your code starts here - everything before this is setup code



    # Create a dataloader with batch size 32 and shuffle set to True

    # Create lists to store the loss and accuracy. These are the metrics we would like to track.

    # Create a loop to iterate through the dataloader. For each batch:

    #     Move the batch to the GPU
    #     Run the forward pass
    #     Calculate the loss
    #     Calculate the accuracy
    #     Average the loss and accuracy across all GPUs using dist.reduce
    #     Append the loss and accuracy to the lists

    # Finally, print the averaged loss and accuracy on rank 0.

    # Remember to destroy the process group at the end of the script. You can use dist.barrier() to ensure that all processes have reached this point before destroying the process group, and dist.destroy_process_group() to destroy the process group.


    # your code ends here - this is followed by teardown code
    dist.barrier()  # wait for all process to reach this point
    dist.destroy_process_group()


if __name__ == '__main__':
    args = argparse.Namespace(cluster_id=0, rank=-1, world_size=WORLD_SIZE)
    if args.rank == -1:
        # we are the parent process, spawn children
        for rank in range(args.cluster_id, TOTAL_RANKS, CLUSTER_SIZE):
            pid = os.fork()
            if pid == 0:
                # child process
                args.rank = rank
                main(args=args)
                break
    # wait for all children to finish
    if args.rank == -1:
        os.waitid(os.P_ALL, 0, os.WEXITED)