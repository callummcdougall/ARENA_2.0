# %% 
import sys
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp
from typing import List
from pathlib import Path
import gdown

gdown.download("https://drive.google.com/file/d/1QgkqHSPDwQD-Z0K0-4CUhp8fW-X0hWds/view", '/tmp/libnccl.so.2.18.1', quiet=False, fuzzy=True)
gdown.download("https://drive.google.com/file/d/1tqUv0OktQdarW8hUyHjqNnxDP1JyUdkq/view?usp=sharing", quiet=False, fuzzy=True)

# Make sure exercises are in the path
chapter = r"chapter3_training_at_scale"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


# %%
from test import test_broadcast_naive

def broadcast_naive(tensor: torch.Tensor, src: int):

    me = dist.get_rank()
    if src!=me:
        dist.recv(tensor, src)
    elif src==me:
        for machine in range(dist.get_world_size()):
            if machine!=me:
                dist.send(tensor, machine)



if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
# %%

from test import test_reduce_naive

def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    me = dist.get_rank()
    operand = tensor.clone()
    if dst==me:
        for machine in range(dist.get_world_size()):
            if machine!=me:
                dist.recv(operand, machine)
                if op==ReduceOp.SUM:
                    tensor = tensor + operand
                if op==ReduceOp.PRODUCT:
                    tensor = tensor*operand
                if op==ReduceOp.MAX:
                    tensor = torch.max(tensor, operand)
                if op==ReduceOp.MIN:
                    tensor = torch.min(tensor, operand)
    elif dist!=me:
        dist.send(tensor, dst)
            


if __name__ == '__main__':
    test_reduce_naive(reduce_naive)



# %%
from test import test_allreduce_naive

def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    reduce_naive(tensor, dst=0, op=op)
    broadcast_naive(tensor, src=0)


if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)