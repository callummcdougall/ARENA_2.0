import argparse
import os
import logging
import time
import random
import string
import tqdm

import torch.distributed as dist
import torch
from torchvision import datasets, transforms, models
from torchvision.io import read_image
from torch.utils.data import Subset, DataLoader, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

CLUSTER_SIZE = 1  # the number of seperate compute nodes we have
WORLD_SIZE = 2  # the number of processes we want to launch - this is usually equal to the number of GPUs we have on this machine
TOTAL_RANKS = CLUSTER_SIZE * WORLD_SIZE
UNIGPU = torch.cuda.device_count() == 1  # remember to use the patched NCCL binary if you are using colab/practicing on a single GPU. You might need to compile https://github.com/pranavgade20/nccl-unigpu if you aren't using colab
print(f"{UNIGPU=}")

# specifically for 2 GPUs

def my_layers(model, rank) -> list:
    """
    list of transformer block numbers that are executed by this rank
    """
    num_total_blocks = len(model.transformer.h)
    num_blocks_per_rank = num_total_blocks / TOTAL_RANKS
    start_block = int(rank * num_blocks_per_rank)
    end_block = int((rank + 1) * num_blocks_per_rank)
    return torch.arange(start_block, end_block)

def set_unused_to_none(model, rank):
    """
    We're doing pipeline parallelism, so layers not used on this GPU should be set to None
    """
    if rank != 0:
        model.transformer.word_embeddings = None
        model.transformer.word_embeddings_layer_norm = None

    layers_to_keep = my_layers(model, rank)
    for layer in range(len(model.transformer.h)):
        if layer not in layers_to_keep:
            model.transformer.h[layer] = None
    
    if rank != TOTAL_RANKS - 1:
        model.ln_f = None


def pipeline_forward(model, input_ids, rank):
    """
    forward pass of the model, but only on the layers that are executed on this GPU
    """
    # annoying attention mask stuff


    if rank == 0:
        x = model.transformer.word_embeddings(input_ids)
        intermediate_resid = model.transformer.word_embeddings_layernorm(x)
    else:
        intermediate_resid = torch.empty(0) # need to be exact size of received tensor?
        dist.recv(tensor=intermediate_resid, src=rank - 1)

    layers_to_keep = my_layers(model, rank)
    for layer in layers_to_keep:
            intermediate_resid = model.transformer.h[layer](intermediate_resid)
    
    if rank == TOTAL_RANKS - 1:
        # Send result back to 0 rank
        dist.send(intermediate_resid, dst=0)
    else:
        dist.send(tensor=intermediate_resid, dst=rank + 1)

    dist.barrier()
    

def main(args):
    rank = args.rank
    world_size = args.world_size
    logging.basicConfig(format=f'[rank {args.rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    logging.warning(f'hello')
    # you can use a variety of backends - nccl is the fastest but requires you to have a GPU; gloo is slower but also sometimes works on CPUs (see https://pytorch.org/docs/stable/distributed.html#backends)
    # dist.init_process_group(backend='nccl', init_method=f'file:///tmp/{"".join(random.choice(string.ascii_letters) for _ in range(10))}', world_size=WORLD_SIZE, rank=args.rank)
    dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:12345', world_size=WORLD_SIZE, rank=args.rank)  # this should be a globally accessible IP
    logging.warning(f'distributed.is_initialized {torch.distributed.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {torch.distributed.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {torch.distributed.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {torch.distributed.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {torch.distributed.is_torchelastic_launched()}')

    time.sleep(1)

    # your code starts here - everything before this is setup code
    # load model
    checkpoint="bigscience/bloomz-560m"
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device='cuda:' + str(0 if UNIGPU else rank))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # file_mappings = json.load(open('./dataset/file_mappings_imagenet.json'))
    # logging.warning("Loading Data:")

    # imagenet_valset = list((lambda k=k: read_image(f'./dataset/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
    # # truncate imagenet_valset to 0.01x its length
    # imagenet_valset = imagenet_valset  #[:int(len(imagenet_valset) * 0.01)]
    # # imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
    # imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
    # imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    # transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224), antialias=None), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    # logging.warning("Transforming Data:")
    # imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]

    # Make dataloader
    # if rank == 0:
    #     data_loader = DataLoader(imagenet_valset, batch_size=32, shuffle=True)
    #     logging.warning(f"Created dataloader {data_loader}")

    # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # losses = []
    # accuracies = []
    # Training loop
    if rank == 0:

        prompt = "On a blazingly hot Sunday morning,"
        prompt = tokenizer.encode(prompt, return_tensors="pt").to(device='cuda:' + str(0 if UNIGPU else rank))
        pipeline_forward(model, prompt, rank)
        x = torch.empty(0)
        dist.recv(tensor=x, src=TOTAL_RANKS - 1)
        logits = model.transformer.lm_head(model.transformer.ln_f(x))
        # sample from logits
        next_token = torch.multinomial(torch.nn.functional.softmax(logits[:, -1, :], dim=-1), num_samples=1)
        next_token = tokenizer.decode(next_token[0])
        logging.warning(next_token)


            # loss = torch.nn.functional.cross_entropy(logits, labels)
            # logging.warning(loss.item())
            # accuracy = (logits.argmax(dim=1) == labels).float().mean()
            # loss.backward()
            # for p in model.parameters():
            #     dist.all_reduce(p.grad)
            #     p.grad /= world_size
            # optimizer.step()

        #     losses.append(loss.item())
        #     accuracies.append(accuracy.item())

        # avg_loss = sum(losses) / len(losses)
        # avg_accuracy = sum(accuracies) / len(accuracies)
        # logging.warning(f"{avg_loss=}, {avg_accuracy=}")

    else:
        pipeline_forward(model, None, rank)


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