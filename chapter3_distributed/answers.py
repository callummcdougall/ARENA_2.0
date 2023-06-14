# %%
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp
from typing import Callable

import threading

# %%
from test import test_broadcast_naive

def broadcast_naive(tensor: torch.Tensor, src: int):
    if dist.get_rank() == src:
        for i in range(dist.get_world_size()):
            if i != dist.get_rank():
                dist.send(tensor, i)
    else:
        dist.recv(tensor, src)


if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
#%%
from test import test_broadcast_tree

def broadcast_tree(tensor: torch.Tensor, src: int):
    curr_mult = 1
    rank_shifted = lambda: (dist.get_rank() + dist.get_world_size() - src) % dist.get_world_size()
    while curr_mult * 2 <= dist.get_world_size():
        if rank_shifted() < curr_mult:
            print(f"{dist.get_rank()} -> {dist.get_rank() + curr_mult}")
            dist.send(tensor, (dist.get_rank() + curr_mult) % dist.get_world_size())
        elif rank_shifted() < curr_mult * 2:
            print(f"{dist.get_rank()} <- {dist.get_rank() - curr_mult}")
            dist.recv(tensor, (dist.get_rank() - curr_mult) % dist.get_world_size())
        curr_mult *= 2
        dist.barrier()

if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)

#%%
from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    to_shifted = lambda i: (i - src) % dist.get_world_size()
    to_orig = lambda i: (i + src) % dist.get_world_size()
    for i in range(1, dist.get_world_size()):
        if to_shifted(dist.get_rank()) == i-1:
            print(f'{dist.get_rank()} | {to_shifted(dist.get_rank())} -> {to_orig(i)}')
            dist.send(tensor, to_orig(i))
        elif to_shifted(dist.get_rank()) == i:
            dist.recv(tensor, to_orig(i-1))
        dist.barrier()

if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)

#%%
from test import test_allreduce_butterfly

def allreduce_butterfly(tensor: torch.Tensor):
    print(f'INIT {dist.get_rank()} {tensor}')
    rank = bin(dist.get_rank())[2:].zfill(len(bin(dist.get_world_size()-1)[2:]))
    buff = torch.empty_like(tensor)
    for i in range(len(rank)):
        partner_rank = rank[:i] + str(1-int(rank[i])) + rank[i+1:]
        partner_rank = int(partner_rank, 2)
        dist.send(tensor, partner_rank)
        dist.recv(buff, partner_rank)
        tensor += buff
    print(f'FINAL {dist.get_rank()} {tensor}')
if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)

# %%
from test import test_reduce_naive

def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    print(f'rank {dist.get_rank()} {tensor}')
    if dist.get_rank() == dst:
        for i in range(dist.get_world_size()):
            if i != dist.get_rank():
                buff = torch.empty_like(tensor)
                dist.recv(buff, i)
                dist.barrier()
                if op == ReduceOp.SUM:
                    tensor += buff
                elif op == ReduceOp.PRODUCT:
                    tensor *= buff
                elif op == ReduceOp.MAX:
                    tensor = torch.max(tensor, buff)
                elif op == ReduceOp.MIN:
                    tensor = torch.min(tensor, buff)
                else:
                    raise NotImplementedError(f'op {op} not implemented')
    else:
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.send(tensor, dst)
            elif i == dst:
                continue
            dist.barrier()
    dist.barrier()



if __name__ == '__main__':
    test_reduce_naive(reduce_naive)

# %%
from test import test_reduce_tree

def reduce_tree():
    pass

if __name__ == '__main__':
    test_reduce_tree(reduce_tree)

# %%
