import os, sys
from pathlib import Path
chapter = r"chapter3_training_at_scale"
instructions_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/instructions").resolve()
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)
import streamlit as st
import st_dependencies

st_dependencies.styling()

import platform
is_local = (platform.processor() != "")

def section_0():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/distributed.png" width="350">


Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.


# [3.2] - Distributed training


## Introduction


Welcome to the exciting world of distributed GPU training using PyTorch!

Training deep learning models can be a computationally intensive task, often requiring substantial time and computational resources. However, distributed GPU training provides a solution to this challenge. By leveraging multiple GPUs, either on a single machine or across a network of machines, the training process can be significantly accelerated.

PyTorch, an open-source machine learning library, is one of the leading tools used for distributed training. Its torch.distributed package offers robust support for distributed computing, enabling users to efficiently harness the power of multiple GPUs.

In distributed GPU training, a model's computations are spread across multiple GPUs. Data parallelism is a common technique used in this process, which involves dividing the input data into chunks and processing each chunk on a different GPU in parallel. The computed gradients from each GPU are then synchronized to update the model parameters.

Distributed GPU training is a valuable skill in the modern era of machine learning and deep learning. Mastering it with PyTorch will unlock new levels of computational power and efficiency for your models, enabling you to tackle larger and more complex problems.


## Content & Learning Objectives


#### 1️⃣ Basics of distributed programming

> ##### Learning objectives
> 
> * Learn the structure of the PyTorch distributed class
> * Understand what a process, thread, and rank is
> * Explore what might cause race conditions
> * Learn about common collective operations
> * Implement broadcast, reduce, and all-reduce
> * Consider the effects of different connection topologies

#### 2️⃣ Data parallelism, DDP

> ##### Learning objectives
> 
> * Load and divide a dataset across multiple processes/GPUs
> * Perform independent parallel forward passes on different batches of data, and aggregate the results
> * Compute gradients and share the means across all GPUs with allreduce

#### 3️⃣ Pipeline parallelism

> ##### Learning objectives
> 
> * Load and divide a model across multiple processes/GPUs
> * Send partial results calculated after a partial forward pass to the next process to continue inference
> * Bonus: divide data into minibatches and minimize gpu idle time

#### 4️⃣ Tensor parallelism

> ##### Learning objectives
> 
> * Understand how parameter tensors are split across GPUs
> * Understand how bias tensors can be partitioned

#### 5️⃣ Bonus

> ##### Learning objectives
> 
> * Implement a backward pass for a pipeline parallel transformer


## Setup


```python
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

```



""", unsafe_allow_html=True)


def section_1():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#processes-threads-and-ranks'>Processes, threads, and ranks</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#processes'>Processes</a></li>
        <li><a class='contents-el' href='#threads'>Threads</a></li>
        <li><a class='contents-el' href='#ranks'>Ranks</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#comparing-nccl-and-alternatives'>Comparing <code>nccl</code> and alternatives</a></li>
    <li class='margtop'><a class='contents-el' href='#race-conditions'>Race conditions</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-test-simulated-race-conditions-on-multiple-threads'><b>Exercise</b> - Test simulated race conditions on multiple threads</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#collective-operations'>Collective operations</a></li>
    <li class='margtop'><a class='contents-el' href='#implementing-collective-operations'>Implementing collective operations</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-broadcast'><b>Exercise</b> - Broadcast</a></li>
        <li><a class='contents-el' href='#exercise-reduce'><b>Exercise</b> - Reduce</a></li>
        <li><a class='contents-el' href='#exercise-all-reduce'><b>Exercise</b> - All-reduce</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 1️⃣ Basics of distributed programming


> ##### Learning objectives
> 
> * Learn the structure of the PyTorch distributed class
> * Understand what a process, thread, and rank is
> * Explore what might cause race conditions
> * Learn about common collective operations
> * Implement broadcast, reduce, and all-reduce
> * Consider the effects of different connection topologies



The PyTorch distributed module, officially known as `torch.distributed`, is a part of the PyTorch library designed to support distributed computing. This module provides the tools and capabilities necessary for the use of multiple computing resources, such as multiple GPUs across multiple nodes, in order to speed up training times and manage larger models and datasets.

The PyTorch distributed module also integrates well with other PyTorch modules and APIs, making it a powerful tool for distributed machine learning tasks. It's designed to be flexible, efficient, and straightforward to use, facilitating scalable and efficient distributed training with PyTorch.

Here, we have also provided a fake distributed class that implements much of the functionality of `torch.distributed` but simulates multiple GPUs using threads. You can explore the implementation as time permits in `test.py`.


## Processes, threads, and ranks

In the context of distributed programming, understanding the difference between processes and threads as well as the concept of ranks for PyTorch distributed GPU programming is essential.

### Processes
A process can be thought of as an instance of a computer program that is being executed. It has its own memory space and is managed independently by the operating system. In PyTorch distributed programming, each process is typically associated with a single computational resource, such as a CPU core or a GPU. Each process can execute its code independently, and they typically communicate with each other using inter-process communication mechanisms, such as sending and receiving messages.

### Threads
Threads are the smallest units of execution within a process. All threads within a process share the same memory space, which allows them to read from and write to the same variables and data structures, facilitating easy communication between threads. However, this shared memory space can lead to issues such as race conditions, which must be managed using locks, semaphores, or other synchronization techniques. In PyTorch, computations on the tensors are multithreaded by default, meaning that operations can use multiple CPU cores for improved performance.

### Ranks
In PyTorch's distributed package, a "rank" is a unique identifier given to each process involved in the distributed computation. This is how one process refers to another when they need to communicate or coordinate in some way. The process with rank 0 is typically considered the "main" process and is often used to coordinate the actions of the other "worker" processes. However, all processes can communicate with each other directly, so this is more of a convention than a strict hierarchy. Rank assignment is generally determined by the order in which the processes are launched. In both `torch.distributed` and our fake distributed class, note that you can access the current rank for an instance `dist` using `dist.get_rank()` which returns an integer from `0` to `world_size-1` where `world_size` is the number of GPUs, threads, or other devices based on the distributed class being used.

These concepts provide the foundation for distributed computing in PyTorch, where multiple processes, each potentially running on different computational resources and containing multiple threads, can work together to perform computations on large datasets or complex models.


## Comparing `nccl` and alternatives

The torch.distributed package in PyTorch supports multiple backends like NCCL (NVIDIA Collective Communications Library) and Gloo to cater to different needs and use-cases in distributed computing. The torch.distributed package (`init_process_group`) can be initialized with one of these backends specified as a parameter, and this backend provides the implementation of data transfer between devices.

`nccl`: Pronounced 'nickel', this is the brainchild of NVIDIA. It is exclusively tailored for NVIDIA GPUs, demonstrating the high level of optimization and specialization for NVIDIA's GPU and products like NVLink and NVSwitch. This backend is designed and optimized specifically for NVIDIA GPUs, and it provides routines that are fundamental to constructing multi-GPU and multi-node deep learning applications. NCCL provides fast inter-GPU communication and is especially beneficial when using multiple GPUs on a single node or across multiple nodes for deep learning tasks. It handles operations like all-reduce, all-gather, reduce, broadcast, etc., on multi-dimensional tensors very efficiently.

`gloo`: Gloo, developed by Facebook, is a library that is equipped to support both CPU and GPU operations. However, Gloo is not specific to NVIDIA GPUs and is a more general-purpose library for distributed computing, and its GPU function significantly trails in speed compared to NVIDIA's NCCL. A key merit of Gloo is its superior error message system, making it useful for debugging before transitioning to NCCL. It's beneficial for tasks where the NCCL backend might not be applicable or optimal, especially for CPU operations.

`mpi`: Meanwhile, MPI, an abbreviation for Message Passing Interface, is not a specific library but an open standard dating back to the 90s. Unlike Gloo and NCCL, MPI is primarily designed for clusters with thousands of CPUs. It won't be applicable in our present context.

## Race conditions

A race condition, in the context of distributed GPU training, is a situation where multiple computing processes or threads are attempting to access and manipulate shared data simultaneously, and the outcome depends on the relative timing of these operations.

For example, imagine two or more GPUs updating a shared model parameter at the same time. If GPU 1 reads the parameter's value, GPU 2 also reads the same value, and then both GPUs perform calculations and attempt to write the result back to the shared parameter memory, a race condition may occur. This is because the value written by the first GPU to finish its calculations could be immediately overwritten by the second GPU, leading to data inconsistency and possibly incorrect results.

To avoid such race conditions, strategies such as locking or synchronization barriers can be used. These strategies ensure that only one process can modify a given memory location at a time.


### Exercise - Test simulated race conditions on multiple threads

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 5 minutes on this exercise.
```

Read the code below, which creates functions to add to and subtract from a shared variable multiple times on two different threads, printing the result to the console. Consider how multiple threads accessing the global `value` variable might affect the output. Will this output be deterministic? Run the code to verify your predictions and feel free to adjust values or add delays to explore how this changes the results.


```python
from threading import Thread

# Add to the global variable
def adder(amount, repeats):
    global value
    for _ in range(repeats):
        value += amount
 
# Subtract from the global variable
def subtractor(amount, repeats):
    global value
    for _ in range(repeats):
        value -= amount
        
def add_and_subtract():
    # Start a thread making additions
    adder_thread = Thread(target=adder, args=(1, 1000000))
    adder_thread.start()
    # Start a thread making subtractions
    subtractor_thread = Thread(target=subtractor, args=(1, 1000000))
    subtractor_thread.start()
    # Wait for both threads to finish
    print('Waiting for threads to finish...')
    adder_thread.join()
    subtractor_thread.join()
    # Print the value
    print(f'Value: {value}')


if __name__ == '__main__':
    value = 0
    add_and_subtract()
```

## Collective operations

Now, we will read about a few common operations for transferring and synchronizing data across multiple threads or GPUs, often known as collective operations.

* Broadcast: In a broadcast operation, data from one device (commonly known as the root device) is distributed to all other devices in a distributed system. This operation ensures that all participating devices share the same copy of the data, starting from one source. In the context of GPUs, this could mean sending a copy of a model's parameters from one GPU to all other GPUs in a multi-GPU system.

* Reduce: A reduce operation involves taking an array of input data from all devices in a distributed system and reducing it to a single result using a specified operation (such as addition, multiplication, finding the maximum, etc.). This operation is performed in a parallel manner across all devices. The result is then stored on a designated device, often called the root device. For instance, a reduce operation on GPUs might involve collecting gradients calculated on each GPU and summing them up to get the final gradient update value.

* All-Reduce: All-reduce is a combination of the reduce and broadcast operations. It first performs a reduction operation using a specified operation (such as addition or finding the maximum) on data from all devices. The resulting value is then broadcast to all devices in the system. This ensures that every device in the distributed system holds the identical result of the reduction. An all-reduce operation is commonly used in multi-GPU systems to ensure all GPUs have the same updated model parameters after performing their individual gradient updates.

You can read more about these operations, as well as other popular collective operations (gather, all-gather, etc.) with diagrams in [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html) as well as [Wikipedia](https://en.wikipedia.org/wiki/Collective_operation).


## Implementing collective operations

Here, we will implement broadcast, reduce, and all-reduce using multiple topologies to explore the efficiency of different implementations. The main building blocks of these exercises will be <code>torch.distributed.send()</code> and <code>torch.distributed.recv()</code>, more informations on these can be found [here](https://pytorch.org/docs/stable/distributed.html#point-to-point-communication)


### Exercise - Broadcast

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10 to 20 minutes on each subexercise.
```

First, let's consider various implementations of broadcast, which transfers data from one rank to all other ranks.

1. Implement broadcast using a naive (one-to-all) topology
2. Implement broadcast using a tree (binary split) topology
3. Implement broadcast using a ring (passing to an adjacent rank) topology

Which topology do you expect would be faster in most settings? Consider pros and cons of this approach.


```python
from test import test_broadcast_naive

def broadcast_naive(tensor: torch.Tensor, src: int):
    pass


if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
```

<details>
<summary>Solution</summary>


```python
def broadcast_naive(tensor: torch.Tensor, src: int):
    # SOLUTION
    if dist.get_rank() == src:
        for i in range(dist.get_world_size()):
            if i != dist.get_rank():
                dist.send(tensor, i)
    else:
        dist.recv(tensor, src)
```
</details>


```python
from test import test_broadcast_tree

def broadcast_tree(tensor: torch.Tensor, src: int):
    pass


if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
```

<details>
<summary>Solution</summary>


```python
def broadcast_tree(tensor: torch.Tensor, src: int):
    # SOLUTION
    curr_mult = 1
    rank_shifted = lambda: (dist.get_rank() - src) % dist.get_world_size()
    while curr_mult * 2 <= dist.get_world_size():
        if rank_shifted() < curr_mult:
            dist.send(tensor, (dist.get_rank() + curr_mult) % dist.get_world_size())
        elif rank_shifted() < curr_mult * 2:
            dist.recv(tensor, (dist.get_rank() - curr_mult) % dist.get_world_size())
        curr_mult *= 2
        dist.barrier()
```
</details>


```python
from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    pass


if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)

```

<details>
<summary>Solution</summary>


```python
def broadcast_ring(tensor: torch.Tensor, src: int):
    # SOLUTION
    to_shifted = lambda i: (i - src) % dist.get_world_size()
    to_orig = lambda i: (i + src) % dist.get_world_size()
    for i in range(1, dist.get_world_size()):
        if to_shifted(dist.get_rank()) == i-1:
            dist.send(tensor, to_orig(i))
        elif to_shifted(dist.get_rank()) == i:
            dist.recv(tensor, to_orig(i-1))
        dist.barrier()
```
</details>


### Exercise - Reduce

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 10-20 minutes on each subexercise.
```

Next, implement the reduce operation to move data from all devices to one device, supporting the following operations based on the input parameter:

* Addition, denoted by `op=ReduceOp.SUM`
* Multiplication, denoted by `op=ReduceOp.PRODUCT`
* Maximum, denoted by `op=ReduceOp.MAX`
* Minimum, denoted by `op=ReduceOp.MIN`

and the following topologies:

1. Implement reduce using a naive (all-to-one) topology
2. Implement reduce using a tree (binary join) topology

Which topology do you expect would be faster in most settings? Consider pros and cons of this approach.


```python
from test import test_reduce_naive

def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    pass


if __name__ == '__main__':
    test_reduce_naive(reduce_naive)

```

<details>
<summary>Solution</summary>


```python
def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    # SOLUTION
    if dist.get_rank() == dst:
        for i in range(dist.get_world_size()):
            if i != dist.get_rank():
                buff = torch.empty_like(tensor)
                dist.recv(buff, i)
                dist.barrier()
                if op == ReduceOp.SUM:
                    tensor += buff
                elif op == ReduceOp.PRODUCT:
                    tensor *= buff
                elif op == ReduceOp.MAX:
                    tensor = torch.max(tensor, buff)
                elif op == ReduceOp.MIN:
                    tensor = torch.min(tensor, buff)
                else:
                    raise NotImplementedError(f'op {op} not implemented')
    else:
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.send(tensor, dst)
            elif i == dst:
                continue
            dist.barrier()
    dist.barrier()
```
</details>


```python
from test import test_reduce_tree

def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    pass


if __name__ == '__main__':
    test_reduce_tree(reduce_tree)

```

<details>
<summary>Solution</summary>


```python
def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    # SOLUTION
    curr_mult = dist.get_world_size() / 2
    rank_shifted = lambda: (dist.get_rank() - dst) % dist.get_world_size()
    while curr_mult >= 1:
        if rank_shifted() < curr_mult:
            buff = torch.empty_like(tensor)
            dist.recv(buff, (dist.get_rank() + curr_mult) % dist.get_world_size())
            if op == ReduceOp.SUM:
                tensor += buff
            elif op == ReduceOp.PRODUCT:
                tensor *= buff
            elif op == ReduceOp.MAX:
                tensor = torch.max(tensor, buff)
            elif op == ReduceOp.MIN:
                tensor = torch.min(tensor, buff)
            else:
                raise NotImplementedError(f'op {op} not implemented')
        elif rank_shifted() < curr_mult * 2:
            dist.send(tensor, (dist.get_rank() - curr_mult) % dist.get_world_size())
        curr_mult /= 2
    dist.barrier()
```
</details>

### Exercise - All-reduce Naive

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-20 minutes on this exercise.
```

Finally, implement the all-reduce operation using the following topologies:

Implement all-reduce using a naive (all-to-one-to-all) topology
    1. Init tensors in respective processes based on the rank
    2. Send all tensors to rank 0, using dist.barrier to ensure synchronization (reduce step)
    3. Send the result from rank 0 process to all process (scatter step)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/q_mermaid.svg" width="400">

```python
from test import test_allreduce_naive

def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    pass


if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)

```

<details>
<summary>Solution</summary>

```python
def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    # SOLUTION
    reduce_naive(tensor, dst=0, op=op)
    broadcast_naive(tensor, src=0)
```
</details>

### Exercise - All-reduce Butterfly

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-20 minutes on this exercise.
```
Implement all-reduce using a butterfly topology as depicted in `(c)` below:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/allreduce_topologies.png">

Which topology do you expect would be faster in most settings? Consider the pros and cons of this approach.

Modify the test cases imported here to run the same operation using `dist.all_reduce` and compare the performance for larger tensors of size 1024x1024. You can also try seeing how these methods differ in speed for different tensor sizes. Which one do you expect to perform better, and why? What happens when the world size (as initialized in the test case) is changed?

```python
from test import test_allreduce_butterfly

def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):
    pass


if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)

```

<details>
<summary>Solution</summary>


```python
def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):
    # SOLUTION
    rank = bin(dist.get_rank())[2:].zfill(len(bin(dist.get_world_size()-1)[2:]))
    buff = torch.empty_like(tensor)
    for i in range(len(rank)):
        partner_rank = rank[:i] + str(1-int(rank[i])) + rank[i+1:]
        partner_rank = int(partner_rank, 2)
        dist.send(tensor.clone(), partner_rank)
        dist.recv(buff, partner_rank)
        if op == ReduceOp.SUM:
            tensor += buff
        elif op == ReduceOp.PRODUCT:
            tensor *= buff
        elif op == ReduceOp.MAX:
            tensor = torch.max(tensor, buff)
        elif op == ReduceOp.MIN:
            tensor = torch.min(tensor, buff)
        else:
            raise NotImplementedError(f'op {op} not implemented')
    dist.barrier()
```
</details>




""", unsafe_allow_html=True)


def section_2():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#advantages-of-data-parallelism'>Advantages of data parallelism</a></li>
    <li class='margtop'><a class='contents-el' href='#disadvantages-of-data-parallelism'>Disadvantages of data parallelism</a></li>
    <li class='margtop'><a class='contents-el' href='#torch-dist-multi-server-setup'><code>torch.dist</code> multi-server setup</a></li>
    <li class='margtop'><a class='contents-el' href='#data-parallel-inference'>Data parallel inference</a></li>
    <li class='margtop'><a class='contents-el' href='#data-parallel-training'>Data parallel training</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 2️⃣ Data parallelism, DDP


> ##### Learning objectives
> 
> * Load and divide a dataset across multiple processes/GPUs
> * Perform independent parallel forward passes on different batches of data, and aggregate the results
> * Compute gradients and share the means across all GPUs with allreduce


Data parallelism is our focus today. While it's not the most efficient method, it's relatively straightforward to grasp and implement, and can be further optimized or combined with other approaches for better performance.

Data parallelism works with N identical GPUs:

* We initialize by copying the same weights to each GPU.
* Then, we divide a batch of size `B` into `N` "minibatches", each of size `M=B//N`. For simplicity, let's assume N evenly divides B.
* Every GPU executes a forward and backward pass on its respective minibatch to calculate the local gradients.
* Lastly, the gradients are synchronized across all GPUs with "all-reduce."

"All-reduce" signifies that each GPU exchanges minibatch gradients until all devices possess the same sum. The summed gradients are equivalent to those obtained from a single forward pass on the full batch B, with certain exceptions for batch normalization and dropout layers.

Batch normalization poses a unique challenge because it normally computes a mean over the full batch, but in this case, each device computes a mean over its minibatch. To replicate dropout across all devices, the random number generator on each device needs careful initialization.

Assuming all special cases are addressed and all GPUs hold an identical sum of gradients, each GPU can independently execute an identical optimizer step, deterministically modifying the parameters to produce the same result. This concludes a single iteration, keeping all the devices synchronized.

## Advantages of data parallelism

The primary advantage is that any model that fits on a single GPU can be adapted to a data parallel version without substantial modifications. As the batch elements are independent (except for batch normalization), devices only need to communicate once per batch to sum their gradients.

In contrast, tensor parallelism and pipeline parallelism necessitate sending activations during both forward and backward passes, requiring clever strategies to reduce the amount of communication.

## Disadvantages of data parallelism

One downside is that the communication between GPUs can become overwhelmed, as all GPUs aim to transmit data simultaneously while summing gradients. This issue can be partially mitigated by sending gradients of the later layers as soon as they're calculated, alternating with computing gradients of the earlier layers, and also by utilizing fast interconnects like NVLink.

If a model can't run on a single GPU even with a minibatch size of 1, data parallelism alone isn't viable; instead, one of the other two methods, potentially in combination with data parallelism, must be employed.

From a memory standpoint, data parallelism is inefficient as it duplicates all parameters N times, along with the optimizer state. This duplication can be lessened by distributing the optimizer state across devices, albeit at the cost of increased communication.

As N increases, the minibatch size B/N becomes too small to fully utilize the GPU. While one can increase the total batch size B to compensate, large batches tend to have worse generalization, setting a problem-dependent limit to B's increase.


## `torch.dist` multi-server setup



Here's a template that implements a naive broadcast algorithm - you'll be using the same setup/teardown code everywhere, so it's worth spending some time here trying to understand what is happening - create a new file called broadcast.py, and run it with `run-on-server.sh broadcast.py`


```python
import argparse
import os
import logging
import time
import random
import string

import torch.distributed as dist
import torch
from torchvision import datasets, transforms, models

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
    dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:12345', world_size=WORLD_SIZE, rank=args.rank)  # this should be a globally accessible IP
    logging.warning(f'distributed.is_initialized {torch.distributed.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {torch.distributed.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {torch.distributed.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {torch.distributed.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {torch.distributed.is_torchelastic_launched()}')

    time.sleep(1)

    # your code starts here - everything before this is setup code

    # method 1 to implement broadcast
    if rank == 0:
        tensor = torch.ones((10, 10), device='cuda:'+str(0 if UNIGPU else rank))
        # import pdb; pdb.set_trace()
        for i in range(1, world_size):
            dist.send(tensor, dst=i)  # send tensor to all other ranks
        logging.warning(f'sent tensor {tensor}')
    else:
        tensor = torch.zeros((10, 10), device='cuda:'+str(0 if UNIGPU else rank))
        dist.recv(tensor, src=0)  # every other rank receives tensoor from rank 0
        logging.warning(f'received tensor {tensor}')

    # method 2 to implement broadcast
    dist.broadcast(tensor, src=0)

    logging.warning(f'tensor {tensor}')

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
```

### Exercise - Data parallel inference

We often have really large datasets and/or models that would take forever to train if you only use one GPU - in cases like this, we like to use multiple GPUs to speed up computation. We will start with implementing the forward pass and calculate the loss using the resnet model you made previously.
1. After initializing the distributed process group, load the model(`resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)`) and the dataset:
    ```python
    file_mappings = json.load(open('/dataset/file_mappings_imagenet.json'))
    logging.warning("Loading Data:")

    imagenet_valset = list((lambda k=k: read_image(f'/dataset/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
    imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
    imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
    imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    logging.warning("Transforming Data:")
    imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
    ```
1. Create a dataloader with batch size 32 and shuffle set to True
1. Create lists to store the loss and accuracy. These are the metrics we would like to track.
1. Create a loop to iterate through the dataloader. For each batch:
    1. Move the batch to the GPU
    1. Run the forward pass
    1. Calculate the loss
    1. Calculate the accuracy
    1. Average the loss and accuracy across all GPUs using [`dist.reduce`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce)
    1. Append the loss and accuracy to the lists

    Finally, print the averaged loss and accuracy on rank 0.
1. Remember to destroy the process group at the end of the script. You can use `dist.barrier()` to ensure that all processes have reached this point before destroying the process group, and `dist.destroy_process_group()` to destroy the process group.
1. Run the script with 2 GPUs and compare the results with the single GPU version. You should see that the loss and accuracy are the same, but the time taken is much shorter. To run on two local GPUs, you can use the following command:
    ```bash
    screen -d -m ./deploy.sh 0.0.0.0 --world-size 2 --cluster-size 1 --cluster-id 0;
    ```
    You can also run this on multiple machines by changing the IP address and cluster id. Make sure to run `screen -r` to see the output of the script:
    ```bash
    screen -d -m ./deploy.sh <ip address> --world-size 1 --cluster-size 2 --cluster-id 0;
    screen -d -m ./deploy.sh <ip address> --world-size 1 --cluster-size 2 --cluster-id 1;
    ```
   Running on Colab should be easier - simply use run.sh `!bash run.sh resnet_fwd.py --world-size 2 --cluster-size 1 --cluster-id 0` and change the world size and cluster id accordingly.
  
You might want to start by copying the setup/teardown template from broadcast.py. Then, follow the instructions above to write a forward pass. Remember to use dist.all_reduce to average loss/accuracy after each minibatch's forward pass before you log it.

From this point onwards in this chapter all code will be executed as individual python files. Also they will need to be executed using the <code>run.sh</code> file like so: <code>run.sh <example.py></code>.

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to 30-40 minutes on this exercise.
```

```python
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

CLUSTER_SIZE = 1  # the number of separate compute nodes we have
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

    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).eval()
    file_mappings = json.load(open('/home/ubuntu/file_mappings_imagenet.json'))
    logging.warning("Loading Data:")

    imagenet_valset = list((lambda k=k: read_image(f'/home/ubuntu/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
    imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
    imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
    imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    logging.warning("Transforming Data:")
    imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]

    time.sleep(1)

    # your code starts here - everything before this is setup code
    
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
```

<details>
<summary>Solution</summary>
 
 
```python 
    
dataloader = DataLoader(imagenet_valset, shuffle=True, batch_size=32, num_workers=4, pin_memory=True, pin_memory_device='cuda:'+str(0 if UNIGPU else rank))
resnet34 = resnet34.to(device='cuda:'+str(0 if UNIGPU else rank))
losses = []
accuracies = []

with torch.no_grad():
    for x, y in dataloader:
        x = x.to(device='cuda:'+str(0 if UNIGPU else rank))
        y = y.to(device='cuda:'+str(0 if UNIGPU else rank))
        y_hat = resnet34(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        # logging.warning(f'loss {loss}')
        dist.reduce(loss, 0, op=dist.ReduceOp.AVG)  # average the loss across all processes
        dist.reduce(accuracy, 0, op=dist.ReduceOp.AVG)  # average the accuracy across all processes
        losses.append(loss.item())
        accuracies.append(accuracy.item())

if rank == 0:
    logging.warning(f'average loss {t.tensor(losses).mean()}')
    logging.warning(f'average accuracy {t.tensor(accuracies).mean()}')
 ```
</details>

### Exercise - Data parallel training

Now that we know how a forward pass through our resnet looks like, we can write a backward pass so that we can train our model faster by sharing the gradient computation across multiple GPUs. This looks a lot like a regular forward pass, so start by writing a training loop that does a forward pass, computes the loss, and then computes the gradient of the loss with respect to the parameters of the model. At this point, we will share the gradients across all GPUs and then update the parameters of the model. After you have calculated the gradients with `loss.backward()`, you can access the gradients of the parameters with `parameter.grad`. use `dist.all_reduce` to average the gradients across all GPUs. You can then update the parameters with `optimizer.step()`.

Optionally, log the loss and accuracy metrics, and see how they improve as you train the model.

If you are training a model from scratch, remember to ensure that all the models have the same weights - this can be done by setting a random seed, or `dist.broadcast()`

```yaml
Difficulty: 🔴🔴🔴🔴🔴
Importance: 🔵🔵🔵🔵🔵

You should spend up to 30-40 minutes on this exercise.
```

```python

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

    logging.basicConfig(format=f'[rank {args.rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    logging.warning(f'hello')
    # you can use a variety of backends - nccl is the fastest but requires you to have a GPU; gloo is slower but also sometimes works on CPUs (see https://pytorch.org/docs/stable/distributed.html#backends)
    # dist.init_process_group(backend='nccl', init_method=f'file:///tmp/{"".join(random.choice(string.ascii_letters) for _ in range(10))}', world_size=WORLD_SIZE, rank=args.rank)
    dist.init_process_group(backend='nccl', init_method=f'tcp://192.9.158.9:12345', world_size=TOTAL_RANKS, rank=args.rank)  # this should be a globally accessible IP
    logging.warning(f'distributed.is_initialized {torch.distributed.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {torch.distributed.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {torch.distributed.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {torch.distributed.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {torch.distributed.is_torchelastic_launched()}')

    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    file_mappings = json.load(open('/home/ubuntu/file_mappings_imagenet.json'))
    logging.warning("Loading Data:")

    imagenet_valset = list((lambda k=k: read_image(f'/home/ubuntu/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
    imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
    imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
    imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    logging.warning("Transforming Data:")
    imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]

    time.sleep(1)

    # your code starts here - everything before this is setup code

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
```

<details>
<summary>Solution</summary>


```python
dataloader = DataLoader(imagenet_valset, shuffle=True, batch_size=256, num_workers=4, pin_memory=True, pin_memory_device='cuda:'+str(0 if UNIGPU else rank))
resnet34 = resnet34.to(device='cuda:'+str(0 if UNIGPU else rank))
resnet34.train()
losses = []
accuracies = []

optim = torch.optim.Adam(resnet34.parameters(), lr=1e-6)

for i in range(args.epochs):
    logging.warning(f'epoch {i}')
    if rank == 0:
        dataloader = tqdm.tqdm(dataloader)
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
            p.grad = p.grad / TOTAL_RANKS  # average the gradients across all processes - alternatively, you can tweak the batch size:learning rate ratio to achieve the same effect

        optim.step()

        accuracy = (y_hat.argmax(1) == y).float().mean()
        dist.reduce(loss, 0, op=dist.ReduceOp.AVG)  # average the loss across all processes
        dist.reduce(accuracy, 0, op=dist.ReduceOp.AVG)  # average the accuracy across all processes
        losses.append(loss.item())
        accuracies.append(accuracy.item())

    if rank == 0:
        logging.warning(f'average loss {t.tensor(losses).mean()}')
        logging.warning(f'average accuracy {t.tensor(accuracies).mean()}')
</details>

""", unsafe_allow_html=True)


def section_3():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#pipeline-parallel-inference'>Pipeline parallel inference</a></li>
    <li class='margtop'><a class='contents-el' href='#bonus-exercises'>Bonus exercises</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 3️⃣ Pipeline parallelism


> ##### Learning objectives
> 
> * Load and divide a model across multiple processes/GPUs
> * Send partial results calculated after a partial forward pass to the next process to continue inference
> * Bonus: divide data into minibatches and minimize gpu idle time


### Exercise - Pipeline parallel inference

Although data parallelism can be used to speed up inference and training in several scenarios, you are still limited by the memory of a single GPU. To work with models that are too large to fit on a single GPU, you have to use pipeline parallelism - splitting the model into several parts, each on a separate GPU. The forward pass is then performed sequentially, with the output of one part being fed into the next part. This allows you to use models that are too large to fit on a single GPU.

Here's a couple of things to consider when using pipeline parallelism:
- The model has to be split into parts that are roughly equal in size. If one part is much larger than the others, it will become the bottleneck.
- Sending data between GPUs is expensive. So, you should try to split up your model in places that minimize the amount of data that needs to be sent between GPUs - for transformers, this is probably between each attention block.
- Ideally, each GPU will get some data, process it, and send it to the next GPU. Avoid a GPU getting data multiple times during a forward pass, as this will cause it to be idle while waiting for data.

The version of pipeline parallelism we are going to implement is fairly basic - split up the model into a bunch of parts depending on the number of GPUs you have (two to start with), and write forward pass that transfers the data between the GPUs whenever required.

1. To start, load the model we will be using for this exercise(`model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")`) on all ranks. If your model is too large to fit in memory (for example, [bloomz-176b](https://huggingface.co/bigscience/bloomz/tree/main)), you will have to dig around the code to figure out which shard corresponds to what weight and assemble the model manually, but that is out of scope for this exercise.
1. In each rank, calculate the layers that rank is responsible for. You should also set the other layers to None to be extra sure that they are not used. Looking at [bloom's forward pass](https://github.com/huggingface/transformers/blob/c5454eba9eac00a3e7d0a46a3d25aacd43187f1e/src/transformers/models/bloom/modeling_bloom.py#L682) might be belpful - I basically copied the forward function, and added `dist.send` and `dist.recv` calls wherever applicable.
1. Start the forward pass by sending the input to the first rank. The first rank should then perform the forward pass on its layers, and send the output to the next rank. The next rank should then perform the forward pass on its layers, and send the output to the next rank. Repeat until you reach the last rank, which should return the output to the first rank (which should then return the output to the user).

```yaml
Difficulty: 🔴🔴🔴🔴🔴
Importance: 🔵🔵🔵🔵

You should spend up to 30-40 minutes on this exercise.
```
<details>
<summary>Solution</summary>

```python
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
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
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
```
</details>

## Bonus exercises

- Implement [key-value caching](https://arena-ch1-transformers.streamlit.app/[1.1]_Transformer_from_Scratch#exercise-implement-caching) to reduce the amount of computation that needs to be done in each forward pass. You can also look at the solution for a way of doing this.
- Write a backward pass and train the model!




""", unsafe_allow_html=True)


def section_4():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#tensor-parallel-linear-layer'>Tensor parallel - Linear layer</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-tensor-parallelism-for-bias-parameter'><b>Exercise</b> - Tensor parallelism for bias parameter</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 4️⃣ Tensor parallelism


> ##### Learning objectives
> 
> * Understand how parameter tensors are split across GPUs
> * Understand how bias tensors can be partitioned



(This content is from MLAB 2.0 curriculum.)

In tensor parallelism, we divide up individual parameter tensors across GPUs. In some cases like the transformer's embedding weight, this could be necessary because the parameter itself is too large to fit on one GPU.

Once divided, each GPU does as much as possible on its own, and we insert the minimum amount of communication between GPUs needed.

Today, we'll build up a tensor parallel implementation of the GPT architecture and use it to perform inference on 4 GPUs simultaneously. We will need tensor parallel versions of these layers that you've previously implemented:

- Linear
- Embedding
- UnidirectionalAttention

To start, we'll test with a simulated multi-GPU setup and then once our code is working, move up to a real machine with multiple GPUs.



## Tensor parallel - Linear layer

A `Linear(in_channels, out_channels)` has a weight matrix of shape `(out_channels, in_channels)`, and the forward method computes $y = x {W}^\intercal + b$.

The fact that the weight is transposed is an implementation detail: it means that columns of ${W}^\intercal$ are contiguous in memory which allows faster multiplication.

We will implement two different methods for splitting up the calculation: partitioning either rows or columns of ${W}^\intercal$ across devices. To be specific, for `Linear(3, 4)` the weight multiplication could look like this:

Partition columns, concatenating results to form output:
$$
\begin{equation*}
\begin{gather*}
\left[
\begin{array}{ccc}
x_0 & x_1 & x_2
\end{array}
\right]
\left[
\begin{array}{cc:cc}
w_{00} & w_{01} & w_{02} & w_{03} \\
w_{10} & w_{11} & w_{12} & w_{13} \\
w_{20} & w_{21} & w_{22} & w_{23} \\
\end{array}
\right] \\
\begin{array}{cc}
\hspace{7.4em}\text{\scriptsize GPU 0}&\hspace{2.2em} \text{\scriptsize GPU 1}
\end{array}
\end{gather*}
=
\begin{gather*}
\left[
\begin{array}{c}
\sum_i w_{i0} x_i \\
\sum_i w_{i1} x_i
\end{array} \\
\right] \\
\left[
\begin{array}{c}
\sum_i w_{i2} x_i \\
\sum_i w_{i3} x_i
\end{array} \\
\right]
\end{gather*}

\begin{array}{c}
\text{\scriptsize GPU 0} \\ \\
\text{\scriptsize GPU 1}
\end{array}

\end{equation*}
$$

Partition rows, adding elementwise to combine contributions:
$$
\begin{equation*}
\begin{gather*}
\begin{array}{c}
\end{array}\\

\left[
\begin{array}{cc:c}
x_0 & x_1 & x_2
\end{array}
\right] \\

\begin{array}{cc}
\hspace{1em}\text{\scriptsize GPU 0} & \text{\scriptsize GPU 1}
\end{array}
\end{gather*}

\left[
\begin{array}{cccc}
w_{00} & w_{01} & w_{02} & w_{03} \\
w_{10} & w_{11} & w_{12} & w_{13} \\
\hdashline
w_{20} & w_{21} & w_{22} & w_{23} \\
\end{array}
\right]

\begin{array}{c}
\\[0.5pt]
\text{\scriptsize GPU 0} \\[4pt]
\text{\scriptsize GPU 1}
\end{array}

=
\begin{gather*}
\left[
\begin{array}{c}
w_{00} x_0 + w_{10} x_1 \\
w_{01} x_0 + w_{11} x_1 \\
w_{02} x_0 + w_{12} x_1 \\
w_{03} x_0 + w_{13} x_1
\end{array} \\
\right] \\
\text{\scriptsize GPU 0}
\end{gather*}
+
\begin{gather*}
\left[
\begin{array}{c}
w_{20} x_2 \\
w_{21} x_2 \\
w_{22} x_2 \\
w_{23} x_2
\end{array} \\
\right] \\
\text{\scriptsize GPU 1}
\end{gather*}
\end{equation*}
$$

In the first scheme, each device needs the full input `x` and is solely responsible for a subset of the output elements. Concatenating all the subsets gives the full output.

In the second scheme, each device can take a partition of `x` and computes partial sums for every output element. Summing all the partial sums gives the full output.


### Exercise - Tensor parallelism for bias parameter

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 5-10 minutes on this exercise.
```

We have described partitioning the weight parameter above. In each scheme, how would you partition the bias parameter?

<details>
<summary>Answer - Partitioning the Bias</summary>

In the first scheme, if we want each rank to have the final output for a subset, then we should partition the bias along the output dimension as well.

In the second scheme, two reasonable ideas are:

- Storing the entire bias on rank 0 and just adding it there before communicating.
- Partitioning the bias and having each rank add their slice at the appropriate offset

The second way distributes the work evenly, but in practice both the storage for the bias and the computation are negligible.

</details>




""", unsafe_allow_html=True)


def section_5():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
        <li><a class='contents-el' href='#exercise-gather-and-all-gather'><b>Exercise</b> - Gather and all-gather</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(r"""

# 5️⃣ Bonus


> ##### Learning objectives
> 
> * Implement a backward pass for a pipeline parallel transformer


## Gather and all-gather

Implement gather and all-gather using naive and tree topologies, based on the description of this function in NCCL documentation linked earlier.




""", unsafe_allow_html=True)


func_page_list = [
    (section_0, "🏠 Home"),     (section_1, "1️⃣ Basics of distributed programming"),     (section_2, "2️⃣ Data parallelism, DDP"),     (section_3, "3️⃣ Pipeline parallelism"),     (section_4, "4️⃣ Tensor parallelism"),     (section_5, "5️⃣ Bonus"), 
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = dict(zip(page_list, range(len(page_list))))

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    idx = page_dict[radio]
    func = func_list[idx]
    func()

page()
