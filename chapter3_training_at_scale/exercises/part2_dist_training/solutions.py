# %%
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp
from typing import List

# %% Race conditions
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
    
# %% Broadcast

from test import test_broadcast_naive

def broadcast_naive(tensor: torch.Tensor, src: int):
    # SOLUTION
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

if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)

#%%
from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    # SOLUTION
    to_shifted = lambda i: (i - src) % dist.get_world_size()
    to_orig = lambda i: (i + src) % dist.get_world_size()
    for i in range(1, dist.get_world_size()):
        if to_shifted(dist.get_rank()) == i-1:
            dist.send(tensor, to_orig(i))
        elif to_shifted(dist.get_rank()) == i:
            dist.recv(tensor, to_orig(i-1))
        dist.barrier()

if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)

# %% Reduce
from test import test_reduce_naive

def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    # SOLUTION
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

def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    # SOLUTION
    curr_mult = dist.get_world_size() / 2
    rank_shifted = lambda: (dist.get_rank() - dst) % dist.get_world_size()
    while curr_mult >= 1:
        if rank_shifted() < curr_mult:
            buff = torch.empty_like(tensor)
            dist.recv(buff, (dist.get_rank() + curr_mult) % dist.get_world_size())
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
        elif rank_shifted() < curr_mult * 2:
            dist.send(tensor, (dist.get_rank() - curr_mult) % dist.get_world_size())
        curr_mult /= 2
    dist.barrier()

if __name__ == '__main__':
    test_reduce_tree(reduce_tree)

#%% All-reduce
from test import test_allreduce_naive

def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    # SOLUTION
    reduce_naive(tensor, dst=0, op=op)
    broadcast_naive(tensor, src=0)

if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)

#%%
from test import test_allreduce_butterfly

def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):
    # SOLUTION
    rank = bin(dist.get_rank())[2:].zfill(len(bin(dist.get_world_size()-1)[2:]))
    buff = torch.empty_like(tensor)
    for i in range(len(rank)):
        partner_rank = rank[:i] + str(1-int(rank[i])) + rank[i+1:]
        partner_rank = int(partner_rank, 2)
        dist.send(tensor.clone(), partner_rank)
        dist.recv(buff, partner_rank)
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
    dist.barrier()

if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)
