
# %%
from IPython import get_ipython

ipython = get_ipython()

ipython.run_line_magic("load_ext", "autoreload")

ipython.run_line_magic("autoreload", "2")

# %%
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

# %%
class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> List[Node]:
    return node.children


def topological_sort(node: Node, get_children: Callable) -> List[Node]:
    '''
    Return a list of node's descendants in reverse topological order from future to past (i.e. `node` should be last).

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    '''
    nodes = [node]
    out_nodes = []
    forbidden = set()

    while nodes:
        node = nodes[0]
        forbidden.add(node)
        children = get_children(node)
        if any(child in nodes for child in children):
            raise RuntimeError("cyclic graph")

        if all(child in out_nodes for child in children):
            out_nodes.append(node)
            nodes.pop(0)
            forbidden.remove(node)
        else:
            nodes = children + nodes
    
    return out_nodes



if MAIN:
    tests.test_topological_sort_linked_list(topological_sort)
    tests.test_topological_sort_branching(topological_sort)
    tests.test_topological_sort_rejoining(topological_sort)
    tests.test_topological_sort_cyclic(topological_sort)
# %%
