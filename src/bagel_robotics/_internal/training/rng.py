"""Rank-local random state: never initialize contexts on other allocated GPUs."""
from contextlib import contextmanager
import random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'])


@contextmanager
def diagnostic_rng():
    state=rng_state()
    try:yield
    finally:restore_rng(state)
