# %%
import collections

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
from transformers import AutoModelForCausalLM, AutoTokenizer
# %%

# %%

assert torch.cuda.device_count() > 0  # make sure we have GPUs

parser = argparse.ArgumentParser(description='ARENA distributed training example')
parser.add_argument('--cluster-id', type=int, default=0, help='cluster id')
parser.add_argument('--cluster-size', type=int, default=2, help='cluster id')
parser.add_argument('--rank', type=int, default=-1, help='rank')
parser.add_argument('--world-size', type=int, default=1, help='world size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--input', type=str, default="Hello", help='Str to pass through the model.')

#args = parser.parse_args()

class args:
    rank = -1
    cluster_size = 1
    world_size = 2
    input_string = "Mary and John went to the park. There, Mary gave a flower to "

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
    dist.init_process_group(backend='nccl', init_method=f'tcp://0.0.0.0:6000', world_size=TOTAL_RANKS, rank=args.rank)  # this should be a globally accessible IP
    logging.warning(f'distributed.is_initialized {torch.distributed.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {torch.distributed.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {torch.distributed.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {torch.distributed.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {torch.distributed.is_torchelastic_launched()}')

    time.sleep(1)

    # code starts here - everything before this is setup code
    #input = args.input
    input = "Mary and John went to the park. There, Mary gave a flower to"
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m").to("cuda")
    base_model = model.base_model

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    input_ids = tokenizer.encode(input, return_tensors="pt").to("cuda")

    # In each rank, calculate the layers that rank is responsible for. Set the other layers to None to be extra sure that they are not used.
    layers_per_rank = len(base_model.h) // WORLD_SIZE

    layers = list(range(len(base_model.h)))
    rank_first_layer = layers_per_rank * rank
    next_rank_first_layer = layers_per_rank * (rank + 1)
    
    for l in (layers[:rank_first_layer] + layers[next_rank_first_layer:]):
        base_model.h[l] = None

    # On rank 0, do embedding
    inputs_embeds = base_model.word_embeddings(input_ids)  
    hidden_states = base_model.word_embeddings_layernorm(inputs_embeds)
    
    batch_size, seq_length = input_ids.shape
    attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
    alibi = base_model.build_alibi_tensor(attention_mask, base_model.num_heads, dtype=hidden_states.dtype)

    head_mask = base_model.get_head_mask(None, base_model.config.n_layer)

    causal_mask = base_model._prepare_attn_mask(
        attention_mask,
        input_shape=(batch_size, seq_length),
        past_key_values_length=0,
    )

    # Do the forward pass rank by rank
    for current_rank in range(WORLD_SIZE):
        if rank == current_rank:
            if rank > 0:
                logging.warning(f"Rank: {rank}, about to receive, HS shape: {hidden_states.shape}")
                dist.recv(hidden_states, rank - 1)
            logging.warning(f"Rank {rank} going to pass input into layers")
            for i, block in enumerate(base_model.h[rank_first_layer:next_rank_first_layer]):
                outputs = block(
                        hidden_states,
                        layer_past=None,
                        attention_mask=causal_mask,
                        head_mask=head_mask[i + rank_first_layer],
                        use_cache=None,
                        output_attentions=base_model.config.output_attentions,
                        alibi=alibi,
                    )
                hidden_states = outputs[0]
            if rank != WORLD_SIZE - 1:
                logging.warning(f"Rank: {rank}, about to send, HS shape: {hidden_states.shape}")
                dist.send(hidden_states, rank + 1)

    if rank == WORLD_SIZE - 1:
        hidden_states = base_model.ln_f(hidden_states)
        hidden_states = model.lm_head(hidden_states)
        logits = hidden_states[0, -1]
        out = logits.argmax().item()
        output = tokenizer.decode(out)
        logging.warning(input + output)


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