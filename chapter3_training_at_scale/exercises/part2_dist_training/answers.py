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
import time
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
    
    if dist.get_rank() == src:
        for rank in range(dist.get_world_size()):
            if rank != src:
                dist.send(tensor, rank)
    else:
        dist.recv(tensor, src)
    dist.barrier()


if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
# %%
from test import test_broadcast_tree

def broadcast_tree(tensor: torch.Tensor, src: int):
    size = dist.get_world_size()
    new_rank = (dist.get_rank() - src) % size

    if new_rank != 0:
        parent = (new_rank-1) // 2
        dist.recv(tensor, (parent + src) % size)

    left_child = 2*new_rank + 1
    right_child = left_child + 1

    if left_child < size:
        dist.send(tensor, (left_child + src) % size)
    if right_child < size:
        dist.send(tensor, (right_child + src) % size)
    dist.barrier()


if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
# %%
from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    size = dist.get_world_size()
    rank = dist.get_rank()

    parent = (rank - 1)%size
    child = (rank + 1)%size

    if rank != src: # if you're not the source, recieve
        dist.recv(tensor, parent)
    if rank != (src-1)%size: # if you're the parent of the source, don't transmit
        dist.send(tensor, child)
    dist.barrier()


if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)
# %%

from test import test_reduce_naive

def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    tensor_recieve = tensor.clone()
    size = dist.get_world_size()
    rank = dist.get_rank()

    if rank != dst:
        dist.send(tensor, dst)
    else:
        for i in range(size):
            if i != rank:
                dist.recv(tensor_recieve, i)
                if op == ReduceOp.SUM:
                    tensor += tensor_recieve
                elif op == ReduceOp.PRODUCT:
                    tensor *= tensor_recieve
                elif op == ReduceOp.MAX:
                    tensor = torch.max(tensor_recieve, tensor)
                elif op == ReduceOp.MIN:
                    tensor = torch.min(tensor_recieve, tensor)
                else:
                    return ValueError

if __name__ == '__main__':
    test_reduce_naive(reduce_naive)

#%%

from test import test_reduce_tree

def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):

    rec = torch.empty_like(tensor)

    def reduce(x,y):
        if op == ReduceOp.SUM:
            return x+y
        elif op == ReduceOp.PRODUCT:
            return x*y
        elif op == ReduceOp.MAX:
            return torch.max(x, y)
        elif op == ReduceOp.MIN:
            return torch.min(x, y)
        else:
            return ValueError

    size = dist.get_world_size()
    new_rank = (dist.get_rank() - dst) % size

    left_child = 2*new_rank + 1
    right_child = 2*new_rank + 2

    if left_child < size:
        dist.recv(rec, (left_child + dst) % size)
        tensor = reduce(tensor, rec)
    if right_child < size:
        dist.recv(rec, (right_child + dst) % size)
        tensor = reduce(tensor, rec)

    if new_rank != 0:
        parent = (new_rank-1) // 2
        dist.send(tensor, (parent + dst) % size)

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
import math

def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):

    rec = torch.empty_like(tensor)

    def reduce(x,y = tensor):
        if op == ReduceOp.SUM:
            tensor = x+y
        elif op == ReduceOp.PRODUCT:
            tensor = x*y
        elif op == ReduceOp.MAX:
            tensor = torch.max(x, y)
        elif op == ReduceOp.MIN:
            tensor = torch.min(x, y)
        else:
            return ValueError
    
    size = dist.get_world_size()
    rank = dist.get_rank()

    max_depth = math.ceil(math.log2(size))
    for d in range(max_depth):
        pair = rank ^ (1<<d)
        if (pair > rank):
            dist.send(tensor, pair)
            dist.recv(rec, pair)
        else:
            dist.recv(rec, pair)
            dist.send(tensor, pair)
        reduce(rec)
        
        # dist.barrier()

if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)
# %%
import json
file_mappings = json.load(open('file_mappings_imagenet.json'))
logging.warning("Loading Data:")

imagenet_valset = list((lambda k=k: read_image(f'val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
logging.warning("Transforming Data:")
imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
#%%


