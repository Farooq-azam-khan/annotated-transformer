import os
import math
import copy  # TODO: looup copy doc
import time
import altair as alt  # TODO: what is this?
import pandas as pd
import spacy
import GPUtil  # TODO: what does this lib do?
import warnings

# TORCH IMPORTS
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad  # TODO: lookup pad doc
from torch.optim.lr_scheduler import LambdaLR  # TODO: how does this work?
from torch.utils.data import DataLoader
from torch.utils.data.distributed import (
    DistributedSampler,
)  # TODO: what is a distrbuted sampler?
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# TODO: review torchtext lib
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets

warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


# https://pytorch.org/docs/stable/optim.html
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encode(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decode(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init_(self, d_model, vocab):
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    return nn.ModuleList(
        [copy.deepcopy(module) for _ in range(N)]
    )  # N identical layers


class LayerNorm(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


if __name__ == "__main__":
    print("Hello World")
