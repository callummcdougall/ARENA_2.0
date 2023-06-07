import time

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

    print(tensor)

# if __name__ == '__main__':
#     test_broadcast_naive(broadcast_naive)
#%%

from test import test_broadcast_tree
def broadcast_tree():
    tensor = torch.tensor([dist.get_rank()], dtype=torch.float32)
    curr_mult = 1
    while curr_mult * 2 < dist.get_world_size():
        print(f"[rank {dist.get_rank()} curr_mult {curr_mult}]")
        if dist.get_rank() < curr_mult:
            print(f"{dist.get_rank()} -> {dist.get_rank() + curr_mult}")
            dist.send(tensor, dist.get_rank() + curr_mult)
        elif dist.get_rank() < curr_mult * 2:
            print(f"{dist.get_rank()} <- {dist.get_rank() - curr_mult}")
            dist.recv(tensor, dist.get_rank() - curr_mult)
        curr_mult *= 2
        dist.barrier()

if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
