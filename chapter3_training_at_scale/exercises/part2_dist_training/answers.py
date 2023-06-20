#%%
from math import ceil, log
import sys
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp
from typing import List
from pathlib import Path
import gdown

gdown.download(
    "https://drive.google.com/file/d/1QgkqHSPDwQD-Z0K0-4CUhp8fW-X0hWds/view",
    '/tmp/libnccl.so.2.18.1',
    quiet=False,
    fuzzy=True)
gdown.download(
    "https://drive.google.com/file/d/1tqUv0OktQdarW8hUyHjqNnxDP1JyUdkq/view?usp=sharing",
    quiet=False,
    fuzzy=True)

# Make sure exercises are in the path
chapter = r"chapter3_training_at_scale"
exercises_dir = Path(
    f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

#%%
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
    adder_thread = Thread(target=adder, args=(1, 1000_000))
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
    # vals = []
    for _ in range(1000):
        value = 0
        add_and_subtract()
        vals.append(value)
    import plotly.express as px
    #%%
    px.histogram(x=vals, nbins=30).show()
# %%
px.line(sorted(vals)).show()
# %%
from test import test_broadcast_naive


def broadcast_naive(tensor: torch.Tensor, src: int):
    world_size = dist.get_world_size()

    if src == dist.get_rank():
        for dst in range(world_size):
            if dst != src:
                dist.send(tensor, dst)
    else:
        dist.recv(tensor, src)


if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
# %%
from test import test_broadcast_tree


def broadcast_tree(tensor: torch.Tensor, src: int):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    def swap_src_0(i: int) -> int:
        if i == 0:
            return src
        if i == src:
            return 0
        return i

    print(src, rank)
    rank = swap_src_0(rank)
    print("new rank", rank)

    if rank != 0:
        recv = swap_src_0((rank - 1) // 2)
        print("recv", recv)
        dist.recv(tensor, recv)

    for dst in (2 * rank + 1, 2 * rank + 2):
        if dst < world_size:
            print("send", dst)
            dist.send(tensor, swap_src_0(dst))


if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
# %%
from test import test_broadcast_ring


def broadcast_ring(tensor: torch.Tensor, src: int):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    prev = (rank - 1) % world_size
    after = (rank + 1) % world_size

    print(rank, src, world_size)
    if rank != src:
        dist.recv(tensor, prev)
    if after != src:
        dist.send(tensor, after)


if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)
# %%
from test import test_reduce_naive


def smash(t1: torch.Tensor, t2, op: ReduceOp):
    if op is ReduceOp.SUM:
        t1 += t2
    elif op is ReduceOp.PRODUCT:
        t1 *= t2
    elif op is ReduceOp.MAX:
        t1.max_(t2)
    elif op is ReduceOp.MIN:
        t1.min_(t2)
    else:
        raise NotImplementedError("meh")


def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank != dst:
        dist.send(tensor, dst)
    else:
        received = torch.empty_like(tensor)
        for src in range(world_size):
            if src != dst:
                dist.recv(received, src)
                smash(tensor, received, op)


if __name__ == '__main__':
    test_reduce_naive(reduce_naive)

# %%
# from test import test_reduce_tree


def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    def swap_dst_0(i: int) -> int:
        if i == 0:
            return dst
        if i == dst:
            return 0
        return i

    print(dst, rank)
    rank = swap_dst_0(rank)

    received = torch.empty_like(tensor)
    for src in (2 * rank + 1, 2 * rank + 2):
        if src < world_size:
            print("send", dst)
            dist.recv(received, swap_dst_0(src))
            smash(tensor, received, op)

    if rank != 0:
        recv = swap_dst_0((rank - 1) // 2)
        print("recv", recv)
        dist.send(tensor, recv)


if __name__ == '__main__':
    test_reduce_tree(reduce_tree)

# %%
from test import test_allreduce_naive


def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    reduce_naive(tensor, 0, op)
    dist.barrier()
    broadcast_naive(tensor, 0)


if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)

# %%
from test import test_allreduce_butterfly
import math


def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    recv = torch.empty_like(tensor)
    for i in range(math.ceil(math.log(world_size, 2))):
        pair = rank ^ (2**i)
        if pair < world_size:
            dist.send(tensor, pair)
            dist.recv(recv, pair)
            smash(tensor, recv, op)


if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)


def osef():
    file_mappings = json.load(open('file_mappings_imagenet.json'))
    logging.warning("Loading Data:")

    mappings = list(file_mappings.items())
    imagenet_valset = [
        (read_image(f'/dataset/val/{k}.JPEG'), int(v))
        for k, v in tqdm(mappings[rank::TOTAL_RANKS], desc=f'[rank {rank}]')
    ]
    imagenet_valset = [(torch.cat([x, x, x], 0) if x.shape[0] == 1 else x, y)
                       for x, y in imagenet_valset]
    transform = torch.jit.script(
        torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])))
    logging.warning("Transforming Data:")
    imagenet_valset = [
        (transform(x), y)
        for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')
    ]
