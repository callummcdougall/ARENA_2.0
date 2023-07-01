# %%
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

# %%
data_root = "/root/dataset"
rank = 0
TOTAL_RANKS=2
UNIGPU=True

resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
file_mappings = json.load(open(f'{data_root}/file_mappings_imagenet.json'))
logging.warning("Loading Data:")

imagenet_valset = list((lambda k=k: read_image(f'{data_root}/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
# imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
imagenet_valset = Subset(imagenet_valset, indices=range(rank, 10, TOTAL_RANKS))
imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
logging.warning("Transforming Data:")
imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]

time.sleep(1)

# your code starts here - everything before this is setup code
imagenet_dataloader = DataLoader(imagenet_valset, batch_size=32, shuffle=True)
loss = []
accuracy = []
print(f"{torch.cuda.device_count()=}")
device = f"cuda:{0 if UNIGPU else rank}"
criterion = t.nn.CrossEntropyLoss()

# In true data parallism I think I would need a copy of the model on each device?
resnet34.to(device)
# %%
for data, labels in imagenet_dataloader:
    data, labels = data.to(device), labels.to(device)

    logits = resnet34(data)

    loss = criterion(logits, labels)
    preds = logits.argmax(dim=-1)

    print(f"{loss=}, {logits.shape=}, {labels.shape=}, {preds.shape=}")
    print(preds, labels, preds == labels)

    accuracy = (preds == labels).sum()

    print(f"{loss.item()=}, {accuracy.item()=}")
    
    break

# %%

# %%
