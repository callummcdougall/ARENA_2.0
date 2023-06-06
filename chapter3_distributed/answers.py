import inspect

import torch
from torch import distributed as dist

from test import test_broadcast


def broadcast_naive():
    tensor = torch.tensor([dist.get_rank()])
    dist.broadcast(tensor, 0)

    print(tensor)

if __name__ == '__main__':
    test_broadcast(broadcast_naive)