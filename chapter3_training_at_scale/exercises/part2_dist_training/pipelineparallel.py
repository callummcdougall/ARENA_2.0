import collections

import tqdm
import argparse
import os
import logging
import time
import random
import string

import torch.distributed as dist
import torch.nn as nn
import torch as t
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from torchvision.io import read_image


assert torch.cuda.device_count() > 0  # make sure we have GPUs

# parser = argparse.ArgumentParser(description='ARENA distributed training example')
# parser.add_argument('--cluster-id', type=int, default=0, help='cluster id')
# parser.add_argument('--cluster-size', type=int, default=2, help='cluster id')
# parser.add_argument('--rank', type=int, default=-1, help='rank')
# parser.add_argument('--world-size', type=int, default=1, help='world size')
# parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

# args = parser.parse_args()

class args:
    rank = -1
    cluster_size =  1
    world_size = 2
    cluster_id = 0



CLUSTER_SIZE = args.cluster_size  # the number of seperate compute nodes we have
WORLD_SIZE = args.world_size  # the number of processes we want to launch - this is usually equal to the number of GPUs we have on this machine
TOTAL_RANKS = CLUSTER_SIZE * WORLD_SIZE
UNIGPU = torch.cuda.device_count() == 1  # remember to use the patched NCCL binary if you are using colab/practicing on a single GPU. You might need to compile https://github.com/pranavgade20/nccl-unigpu if you aren't using colab

def main(args):
    rank = args.rank

    logging.basicConfig(format=f'[rank {args.rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    logging.warning(f'hello')
    # you can use a variety of backends - nccl is the fastest but requires you to have a GPU; gloo is slower but also sometimes works on CPUs (see https://pytorch.org/docs/stable/distributed.html#backends)
    # dist.init_process_group(backend='nccl', init_method=f'file:///tmp/{"".join(random.choice(string.ascii_letters) for _ in range(10))}', world_size=WORLD_SIZE, rank=args.rank)
    dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:12345', world_size=TOTAL_RANKS, rank=args.rank)  # this should be a globally accessible IP
    logging.warning(f'distributed.is_initialized {dist.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {dist.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {dist.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {dist.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {dist.is_torchelastic_launched()}')

    #CODE GO HERE

    # %%

    device='cuda:'+str(0 if UNIGPU else rank)

    import torch
    import torchvision.models as models
    from torchinfo import summary

    resnet34 = models.resnet34(pretrained=True).to(device)

    class PipeNet34(torch.nn.Module):
        def __init__(self):
            super(PipeNet34, self).__init__()
            self.resnet = models.resnet34(pretrained=True)
            
        def forward(self, x):
            # Run each layer individually and return intermediate outputs
        
            okay_buddy = torch.empty((100, 128, 28, 28), device=device)

            if rank == 0:            

                x = self.resnet.conv1(x)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                
                x = self.resnet.maxpool(x)
                
                x = self.resnet.layer1(x)
                
                x = self.resnet.layer2(x)

                dist.send(x, 1)
                print("transmitting...")
                return
            
            else:
            
                dist.recv(okay_buddy, 0)
                print("recieved!")
                x = self.resnet.layer3(okay_buddy)
                
                x = self.resnet.layer4(x)

                x = self.resnet.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.resnet.fc(x)
                return x

    torch.manual_seed(42)
    x = torch.randn((100,3,224,224), device=device)
    pipenet34 = PipeNet34().to(device)
    y1 = pipenet34(x)
    if rank == 1:
        print(y1)
        
        y2 = resnet34(x)
        print(y2)

        print(f"max: {torch.max(y1-y2)}")



# %%




    # your code ends here - this is followed by teardown code
    dist.barrier()  # wait for all process to reach this point
    dist.destroy_process_group()


if __name__ == '__main__':
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