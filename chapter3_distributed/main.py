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


CLUSTER_SIZE = 1  # the number of seperate compute nodes we have
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

    file_mappings = json.load(open('home/ubuntu/file_mappings_imagenet.json'))
    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    logging.warning("Loading Data:")

    imagenet_valset = [(read_image(f'/home/ubuntu/ILSVRC/Data/CLS-LOC/val/{k}.JPEG'), int(v)) for k, v in tqdm.tqdm(list(file_mappings.items()))]
    imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    logging.warning("Transforming Data:")
    imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset)]
    imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))

    # imagenet_valset = list((lambda: read_image(f'/home/ubuntu/ILSVRC/Data/CLS-LOC/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
    # imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
    # imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, position=rank, desc=f'[rank {rank}]')]
    # imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    # transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    # logging.warning("Transforming Data:")
    # imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, position=rank, desc=f'[rank {rank}]')]

    time.sleep(1)

    # your code starts here - everything before this is setup code
    dataloader = DataLoader(imagenet_valset, shuffle=True, batch_size=128, num_workers=4, pin_memory=True, pin_memory_device='cuda:'+str(0 if UNIGPU else rank))
    resnet34 = resnet34.to(device='cuda:'+str(0 if UNIGPU else rank))
    resnet34.train()
    losses = []
    accuracies = []

    optim = torch.optim.Adam(resnet34.parameters(), lr=1e-3)

    for x, y in dataloader:
        resnet34.zero_grad()
        # optim.zero_grad()  # what's the difference?

        x = x.to(device='cuda:'+str(0 if UNIGPU else rank))
        y = y.to(device='cuda:'+str(0 if UNIGPU else rank))
        y_hat = resnet34(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        loss.backward()

        for p in resnet34.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)  # sum the gradients across all processes
            p.grad = p.grad / world_size  # average the gradients across all processes - alternatively, you can tweak the batch size:learning rate ratio to achieve the same effect

        optim.step()

        accuracy = (y_hat.argmax(1) == y).float().mean()
        dist.reduce(loss, 0, op=dist.ReduceOp.AVG)  # average the loss across all processes
        dist.reduce(accuracy, 0, op=dist.ReduceOp.AVG)  # average the accuracy across all processes
        logging.warning(f'rank {rank} loss {loss.item()} accuracy {accuracy.item()}')
        losses.append(loss.item())
        accuracies.append(accuracy.item())

    if rank == 0:
        logging.warning(f'average loss {t.tensor(losses).mean()}')
        logging.warning(f'average accuracy {t.tensor(accuracies).mean()}')

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