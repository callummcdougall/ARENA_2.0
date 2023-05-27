#%%
import os
import sys
import re
import time
import torch as t
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Iterable, Optional, Union, Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_backprop"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

import part5_backprop.tests as tests
from part5_backprop.utils import visualize, get_mnist
from plotly_utils import line

MAIN = __name__ == "__main__"

#%%
class Node:
    def __init__(self, *children):
        self.children = list(children)

def get_children(node: Node) -> List[Node]:
    return node.children

def add_d2n(d, node, depth):
    if depth not in d:
        d[depth] = set([node])
    else:
        d[depth].add(node)


def topological_sort(node: Node, get_children: Callable) -> List[Node]:
    '''
    Return a list of node's descendants in reverse topological order from future to past (i.e. `node` should be last).

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    '''
    depth2nodes = {0: set([node])}  # {depth: node}
    node2depth = {node: 0}
    stack = [node]

    while stack:
        node = stack.pop(-1)
        depth = node2depth[node]
        for c in get_children(node):
            visited = c in node2depth
            if not visited:
                stack.append(c)
                node2depth[c] = depth + 1
            elif visited and depth + 1 > node2depth[c]:
                node2depth[c] = depth + 1
    
    depth2nodes = {}
    for n, d in node2depth.items():
        add_d2n(depth2nodes, n, d)

    sorted_nodes = []
    for i in range(len(depth2nodes)):
        nodes = list(depth2nodes[i])
        sorted_nodes.extend(nodes)
    sorted_nodes = sorted_nodes[::-1]

    return sorted_nodes


if MAIN:
    tests.test_topological_sort_linked_list(topological_sort)
    tests.test_topological_sort_branching(topological_sort)
    tests.test_topological_sort_rejoining(topological_sort)
    tests.test_topological_sort_cyclic(topological_sort)

#%%