from typing import Optional, Tuple, Union

import tqdm
import argparse
import os
import logging

import torch.distributed as dist

import torch as t
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

assert torch.cuda.device_count() > 0  # make sure we have GPUs

parser = argparse.ArgumentParser(description='ARENA distributed training example')
parser.add_argument('--cluster-id', type=int, default=0, help='cluster id')
parser.add_argument('--cluster-size', type=int, default=2, help='cluster id')
parser.add_argument('--rank', type=int, default=-1, help='rank')
parser.add_argument('--world-size', type=int, default=1, help='world size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

args = parser.parse_args()


CLUSTER_SIZE = args.cluster_size  # the number of seperate compute nodes we have
WORLD_SIZE = args.world_size  # the number of processes we want to launch - this is usually equal to the number of GPUs we have on this machine
TOTAL_RANKS = CLUSTER_SIZE * WORLD_SIZE
UNIGPU = torch.cuda.device_count() == 1  # remember to use the patched NCCL binary if you are using colab/practicing on a single GPU. You might need to compile https://github.com/pranavgade20/nccl-unigpu if you aren't using colab


def main(args):
    rank = args.rank
    device = 'cuda:'+str(0 if UNIGPU else rank)

    checkpoint = "bigscience/bloomz-560m"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    logging.basicConfig(format=f'[rank {args.rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    logging.warning(f'hello')
    # you can use a variety of backends - nccl is the fastest but requires you to have a GPU; gloo is slower but also sometimes works on CPUs (see https://pytorch.org/docs/stable/distributed.html#backends)
    # dist.init_process_group(backend='nccl', init_method=f'file:///tmp/{"".join(random.choice(string.ascii_letters) for _ in range(10))}', world_size=WORLD_SIZE, rank=args.rank)
    dist.init_process_group(backend='nccl', init_method=f'tcp://0.0.0.0:12344', world_size=TOTAL_RANKS, rank=args.rank)  # this should be a globally accessible IP

    if rank == 0:
        tmp = t.tensor([1.0]).cuda()
        dist.send(tmp, 1)
    elif rank == 1:
        tmp = t.tensor([2.0]).cuda()
        dist.recv(tmp, 0)


    logging.warning(f'distributed.is_initialized {torch.distributed.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {torch.distributed.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {torch.distributed.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {torch.distributed.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {torch.distributed.is_torchelastic_launched()}')

    start_shard = rank * len(model.transformer.h) // TOTAL_RANKS
    end_shard = (rank + 1) * len(model.transformer.h) // TOTAL_RANKS

    shards_map = {}
    for r in range(TOTAL_RANKS):
        for i in range(r * len(model.transformer.h) // TOTAL_RANKS, (r + 1) * len(model.transformer.h) // TOTAL_RANKS):
            shards_map[i] = r

    for i in range(len(model.transformer.h)):
        if shards_map[i] != rank:
            ref = model.transformer.h[i]
            model.transformer.h[i] = None
            del ref
        else:
            model.transformer.h[i] = model.transformer.h[i].to(device=device)

    # if rank == 0:
    model = model.to(device=device)
    # model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device=device)
        # model.transformer.word_embeddings = model.transformer.word_embeddings.to(device=device)
    def forward(
        model,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else model.config.use_cache
        return_dict = return_dict if return_dict is not None else model.config.use_return_dict

        if rank == 0:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            tmp = t.tensor([batch_size, seq_length], dtype=t.float, device="cuda:0")
        else:
            tmp = t.tensor([0, 0], dtype=t.float, device="cuda:0")
        dist.broadcast(tmp, 0)  # using https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.set is probably better here
        batch_size, seq_length = int(tmp[0].item()), int(tmp[1].item())

        if past_key_values is None:
            past_key_values = tuple([None] * len(model.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = model.get_head_mask(head_mask, model.config.n_layer)

        # if attention_mask is not None:
        if inputs_embeds is None:
            if rank == 0:
                inputs_embeds = model.word_embeddings(input_ids)
            else:
                inputs_embeds = t.zeros((batch_size, seq_length, model.config.hidden_size), dtype=t.float, device=device)
        hidden_states = model.word_embeddings_layernorm(inputs_embeds)


        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if rank == 0:
            tmp = t.tensor([past_key_values_length, seq_length_with_past], dtype=t.long, device=device)
        else:
            tmp = t.tensor([0, 0], dtype=t.long, device=device)
        dist.broadcast(tmp, 0)  # using https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.set is probably better here
        past_key_values_length, seq_length_with_past = tmp[0].item(), tmp[1].item()

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=device)
        else:
            attention_mask = attention_mask.to(device=device)
        dist.broadcast(attention_mask, 0)

        alibi = model.build_alibi_tensor(attention_mask, model.num_heads, dtype=t.float32)

        causal_mask = model._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        hidden_states = hidden_states.to(device=device)

        for i, (block, layer_past) in enumerate(zip(model.h, past_key_values)):
            if output_hidden_states and shards_map[i] != 0:
                dist.send(hidden_states, dst=0)
            if output_hidden_states and rank == 0:
                if shards_map[i] != 0:
                    dist.recv(hidden_states, src=shards_map[i])
                all_hidden_states = all_hidden_states + (hidden_states,)

            # if model.gradient_checkpointing and model.training:
            #
            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             # None for past_key_value
            #             return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)
            #
            #         return custom_forward
            #
            #     outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(block),
            #         hidden_states,
            #         alibi,
            #         causal_mask,
            #         layer_past,
            #         head_mask[i],
            #     )
            # else:
            if shards_map[i] == rank:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

                hidden_states = outputs[0]
            if use_cache is True:
                if rank == 0 and shards_map[i] == 0:
                    presents = presents + (outputs[1],)
                elif rank == 0 and shards_map[i] != 0:
                    presents = presents + ((t.zeros_like(presents[-1][0], device=device).contiguous(),t.zeros_like(presents[-1][1], device=device).contiguous()),)
                    dist.recv(presents[-1][0], src=shards_map[i])
                    dist.recv(presents[-1][1], src=shards_map[i])
                elif rank != 0 and shards_map[i] == rank:
                    presents = presents + (outputs[1],)
                    dist.send(presents[-1][0].contiguous(), dst=0)
                    dist.send(presents[-1][1].contiguous(), dst=0)
                else:
                    presents = presents + (None,)

            if output_attentions:
                if rank == 0 and shards_map[i] == 0:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                elif rank == 0 and shards_map[i] != 0:
                    all_self_attentions = all_self_attentions + (t.zeros_like(outputs[2 if use_cache else 1]),)
                    dist.recv(all_self_attentions[-1], src=shards_map[i])
                elif rank != 0 and shards_map[i] == rank:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                    dist.send(all_self_attentions[-1], dst=0)
                else:
                    all_self_attentions = all_self_attentions + (None,)

                # if rank == 0 and shards_map[i] != 0:
                #     all_self_attentions = all_self_attentions + (t.zeros_like(all_self_attentions[-1], device=device),)
                #     dist.recv(all_self_attentions[-1], src=shards_map[i])
                # elif rank != 0:
                #     dist.send(all_self_attentions[-1], dst=0)

            # move hidden states to next shard
            if (i+1) in shards_map and shards_map[i+1] != shards_map[i]:
                if rank == shards_map[i]:
                    dist.send(hidden_states, dst=shards_map[i+1])
                elif rank == shards_map[i+1]:
                    hidden_states = t.zeros((batch_size, seq_length, model.config.hidden_size), device=device)
                    dist.recv(hidden_states, src=shards_map[i])
        # Add last hidden state
        if rank == 0:
            dist.recv(hidden_states, src=TOTAL_RANKS-1)
            hidden_states = model.ln_f(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        elif rank == TOTAL_RANKS-1:
            dist.send(hidden_states, dst=0)


        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    tokenized = tokenizer("hello there, I am", return_tensors="pt")
    tokens = tokenized.input_ids.to(device=device)
    input_ids = t.zeros((1, 100), dtype=t.long, device=device)
    input_ids[0, :tokens.shape[1]] = tokens
    past_key_values = None
    for i in range(1, 100):
        if rank == 0:
            ret = forward(model.transformer, input_ids=input_ids[:,i-1:i], attention_mask=t.ones_like(input_ids[:,:i]), past_key_values=past_key_values)
            past_key_values = ret.past_key_values
            token = t.distributions.Categorical(logits=model.lm_head(ret.last_hidden_state)).sample()
            if i > tokens.shape[-1]:
                input_ids[0, i] = token.item()
            logging.warning(tokenizer.decode(token.item()))
        else:
            ret = forward(model.transformer, input_ids=input_ids[:,i-1:i], attention_mask=t.ones_like(input_ids[:,:i]), past_key_values=past_key_values)
            past_key_values = ret.past_key_values

    import time
    time.sleep(10)


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
