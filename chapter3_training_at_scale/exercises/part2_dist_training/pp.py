import argparse
import os
import logging
import time
import random
import string

import torch.distributed as dist
import torch
from torchvision import datasets, transforms, models

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bloom import BloomModel

CLUSTER_SIZE = 1  # the number of seperate compute nodes we have
WORLD_SIZE = 2  # the number of processes we want to launch - this is usually equal to the number of GPUs we have on this machine
TOTAL_RANKS = CLUSTER_SIZE * WORLD_SIZE
UNIGPU = torch.cuda.device_count() == 1  # remember to use the patched NCCL binary if you are using colab/practicing on a single GPU. You might need to compile https://github.com/pranavgade20/nccl-unigpu if you aren't using colab


class args:
    rank = -1
    world_size = 2
    epochs = 10
    cluster_id = 0
    cluster_size = 1


class PipedBloom:
    def __init__(self, bloom_model):
        self.model = bloom_model

    def set_piped_modules(self, rank):
        modules_per_rank = self.model.config.n_layer // TOTAL_RANKS
        self.rank_to_layers = {}

        # set all modules to None except those used by rank (as a sanity check)
        for i in range(self.model.config.n_layer):
            if i < rank*modules_per_rank or i >= (rank+1)*modules_per_rank:
                self.model.h[i] = None 

            # modules in [ rank*modules_per_rank, (rank+1)*modules_per_rank ) are left


        # for i in range(TOTAL_RANKS):
        #     self.rank_to_layers[i] = list(range((i)*modules_per_rank, (i+1)*modules_per_rank))


    def forward(self, rank, input_ids, attention_mask): 
        modules_per_rank = self.model.config.n_layer // TOTAL_RANKS

        # if rank == 0:
        inputs_embeds = self.model.transformer.word_embeddings(input_ids)

        hidden_states = self.model.transformer.word_embeddings_layernorm(inputs_embeds) # defined in every rank lol

        alibi = self.model.transformer.build_alibi_tensor(attention_mask, self.model.config.n_head, dtype=hidden_states.dtype)
        batch_size, seq_length = input_ids.shape

        past_key_values = tuple([None] * len(self.model.transformer.h))
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        causal_mask = self.model.transformer._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            # past_key_values_length=past_key_values_length,
            past_key_values_length=past_key_values_length
        )

        head_mask = self.model.transformer.get_head_mask(None, self.model.config.n_layer)

        for i, layer_past in zip(range(self.model.config.n_layer), past_key_values):
            if i < rank*modules_per_rank or i >= (rank+1)*modules_per_rank: # if modules are not used on this rank
                continue 

            if i == rank*modules_per_rank:
                if rank != 0:
                    dist.recv(hidden_states, rank-1)

            # use forward
            hidden_states += self.model.transformer.h[i](hidden_states, head_mask = head_mask, layer_past = layer_past, alibi = alibi, attention_mask=causal_mask)[0]

            if i == (rank+1)*modules_per_rank - 1:
                if rank != TOTAL_RANKS-1:
                    dist.send(hidden_states, rank+1)
                else:
                    dist.send(hidden_states, 0) # send back to rank 0

        # dist.barrier()

        if rank == 0: # do final layernorm and projection
            dist.recv(hidden_states, TOTAL_RANKS-1)
            hidden_states = self.model.transformer.ln_f(hidden_states)
            
            return self.model.lm_head(hidden_states)


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
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m").to(device='cuda:'+str(0 if UNIGPU else rank))
    
    piped_model = PipedBloom(model)
    test_str = ["John and Mary went to the store. John bought a drink and gave it to"]
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    input_ids, attention_mask = tokenizer(test_str).input_ids, tokenizer(test_str).attention_mask
    input_ids =  torch.tensor(input_ids, device='cuda:'+str(0 if UNIGPU else rank))
    attention_mask = torch.tensor(attention_mask, device='cuda:'+str(0 if UNIGPU else rank))
    # print(test_tokens)
    # print(model(torch.tensor(test_tokens, device='cuda:'+str(0 if UNIGPU else rank))))
    logits = piped_model.forward(rank, input_ids, attention_mask)

    print("done with forward")
    dist.barrier()
    if rank == 0:
        print(logits)
        print(tokenizer.batch_decode(torch.argmax(logits, dim=-1)))

        non_piped_logits = model(input_ids)
        print(f"{non_piped_logits=}")
        print(f"non piped decoded is {tokenizer.batch_decode(torch.argmax(non_piped_logits, dim=-1))}")

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