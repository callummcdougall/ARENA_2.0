import torch
from torch import distributed as dist


#%%
from test import test_broadcast_naive
def broadcast_naive():
    tensor = torch.tensor([dist.get_rank()])
    if dist.get_rank() == 0:
        for i in range(dist.get_world_size()):
            if i != dist.get_rank():
                dist.send(tensor, i)
    else:
        dist.recv(tensor, 0)


if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
#%%

from test import test_broadcast_tree
def broadcast_tree():
    tensor = torch.tensor([dist.get_rank()], dtype=torch.float32)
    curr_mult = 1
    while curr_mult < dist.get_world_size():
        if dist.get_rank() < curr_mult:
            dist.send(tensor, dist.get_rank() + curr_mult)
        elif dist.get_rank() < curr_mult * 2:
            dist.recv(tensor, dist.get_rank() - curr_mult)
        curr_mult *= 2
        dist.barrier()

if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
#%%
from test import test_broadcast_ring

def broadcast_ring():
    tensor = torch.tensor([dist.get_rank()], dtype=torch.float32)
    for i in range(1, dist.get_world_size()):
        if dist.get_rank() == i-1:
            dist.send(tensor, i)
        elif dist.get_rank() == i:
            dist.recv(tensor, i-1)
        dist.barrier()

if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)

#%%
from test import test_allreduce_butterfly
def allreduce_butterfly():
    tensor = torch.tensor([dist.get_rank()], dtype=torch.float32)
    rank = bin(dist.get_rank())[2:].zfill(len(bin(dist.get_world_size()-1)[2:]))
    buff = torch.empty_like(tensor)
    for i in range(len(rank)):
        partner_rank = rank[:i] + str(1-int(rank[i])) + rank[i+1:]
        partner_rank = int(partner_rank, 2)
        dist.send(tensor, partner_rank)
        dist.recv(buff, partner_rank)
        tensor += buff

if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)

