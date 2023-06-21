# %%
import sys
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp
from typing import List
from pathlib import Path
import gdown
import traceback
from test import test_reduce_naive
from operator import add, mul
import traceback

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
from test import test_broadcast_naive

def broadcast_naive(tensor: torch.Tensor, src: int):
    world_size = dist.get_world_size()

    if dist.get_rank() == src:
        for rank in range(world_size):
            if rank == src:
                continue

            dist.send(tensor, rank)
    else:
        dist.recv(tensor, src)


if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
# %%
from test import test_broadcast_tree

def broadcast_tree(tensor: torch.Tensor, src: int):
    pass

if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
# %%
from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    src_parent = (src - 1) % world_size

    parent = (rank - 1) % world_size
    child = (rank + 1) % world_size

    if rank != src:
        dist.recv(tensor, parent)

    if rank != src_parent:
        dist.send(tensor, child)

    dist.barrier()

if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)
# %%
from test import test_reduce_naive
from operator import add, mul
import traceback


def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    operations = {
        ReduceOp.SUM:  add,
        ReduceOp.PRODUCT: mul,
        ReduceOp.MAX: max,
        ReduceOp.MIN: min
    }

    accumulate = operations[op]
    rank = dist.get_rank()

    if rank != dst:
        dist.send(tensor, dst)

    if rank == dst:
        buffer = torch.zeros_like(tensor)

        for sender in range(dist.get_world_size()):
            if sender == rank:
                continue

            dist.recv(buffer, sender)
            tensor = accumulate(tensor, buffer)
    


if __name__ == '__main__':
    test_reduce_naive(reduce_naive)
# %%
from test import test_reduce_tree
from math import log2

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    rank = dist.get_rank()
    print(f"{rank}: {tensor}")
    dist.barrier()

    try:
        operations = {
            ReduceOp.SUM:  add,
            ReduceOp.PRODUCT: mul,
            ReduceOp.MAX: max,
            ReduceOp.MIN: min
        }
        accumulate = operations[op]

        world_size = dist.get_world_size()
        assert is_power_of_two(world_size), "world size must be a power of 2"

        shift = lambda x: (x - dst) % world_size
        unshift = lambda x: (x + dst) % world_size

        rank = dist.get_rank()
        if rank == dst:
            dist.barrier()
        
        shifted = shift(rank)
        print(f"{rank} --> {shifted}")

        dist.barrier()

        active = True
        passes = int(log2(world_size))
        for k in range(passes):
            if shifted == 0:
                print(f"===== {shifted} ({rank}): Pass {k + 1} of {passes}")
            dist.barrier()

            if active:
                if shifted % (2**(k+1)):
                    send_to = shifted - 2**k
                    print(f"{shifted} ({rank}): sending {tensor} to {send_to} ({unshift(send_to)})")
                    dist.send(tensor, unshift(send_to))
                    print(f"{shifted} ({rank}): sending complete. deactivating")
                    active = False
                else:
                    receive_from = shifted + 2**k
                    print(f"{shifted} ({rank}): receiving from {receive_from} ({unshift(receive_from)})")
                    buffer = torch.zeros_like(tensor)
                    dist.recv(buffer, unshift(receive_from))
                    print(f"{shifted} ({rank}): received {buffer} from {receive_from}; adding to {tensor}")
                    tensor = accumulate(buffer, tensor)
                    print(f"{shifted} ({rank}): tensor is now {tensor}")

            print(f"{shifted} ({rank}): at barrier")
            dist.barrier()
            print(f"{shifted} ({rank}): past barrier")

        if rank == dst:
            print(f"{shifted} ({rank}): got to end of loop")

        print(f"{shifted} ({rank}): exited")
    except Exception:
        print(traceback.format_exc())

if __name__ == '__main__':
    test_reduce_tree(reduce_tree, dst_rank=3, world_size=4)
     
# %%
def foo():
    passes = 3
    ranks = 8
    active = { _: True for _ in range(ranks) }
    print(f"{active=}")

    for k in range(passes):
        print(f"pass {k}")

        for rank in range(ranks):
            if active[rank]:
                if rank % (2 ** (k + 1)):
                    send_to = rank - 2**k
                    print(f"{rank} sending to {send_to}")
                    print(f"deactivating {rank}")
                    active[rank] = False
                else:
                    receive_from = rank + 2**k
                    print(f"{rank} receiving from {receive_from}")

        print(f"{active=}")

foo()
# %%

world_size = 8
world = list(range(world_size))
dst = 5

shift = lambda x: (x - dst) % world_size
unshift = lambda x: (x + dst) % world_size

shifted = [shift(w) for w in world]
unshifted = [unshift(w) for w in shifted]

print(world)
print(shifted)
print(unshifted)


from test import test_allreduce_naive
from solutions import reduce_naive, broadcast_naive

def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    reduce_naive(tensor, 0, op=op)
    dist.barrier()

    broadcast_naive(tensor, 0)
    


if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)
# %%
def flip_kth_bit(num: int, k: int) -> int:
    mask = 1 << k
    flipped = num ^ mask
    return flipped


from test import test_allreduce_butterfly

def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):
    operations = {
        ReduceOp.SUM:  add,
        ReduceOp.PRODUCT: mul,
        ReduceOp.MAX: max,
        ReduceOp.MIN: min
    }
    accumulate = operations[op]

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    steps = int(log2(world_size))
    for step in range(steps):
        partner = flip_kth_bit(rank, step)
        dist.send(tensor, partner)

        buffer = torch.zeros_like(tensor)
        dist.recv(buffer, partner)

        tensor = accumulate(tensor, buffer)
        dist.barrier()

    print(tensor)

if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)
# %%
