"""Exact restart/rollback and measured unique-window coverage."""
import copy
import random
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
import numpy as np
import torch
from vpp2.full_mask.panel import digest,file_hash

def identity(config,panel):
    source=Path(__file__).resolve().parents[3]
    return digest(dict(config=config,panel=panel['identity'],
        files={str(p.relative_to(source)):file_hash(p) for p in sorted((source/'src/vpp2').rglob('*.py'))}))

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])

@contextmanager
def diagnostic_rng():
    old=rng_state()
    try:yield
    finally:restore_rng(old)

def snapshot(controller,optimizer,cursor,lineage_update,coverage,pruning_updates):
    return dict(controller=copy.deepcopy(controller.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
        cursor=cursor,lineage_update=lineage_update,coverage=dict(coverage),pruning_updates=pruning_updates,rng=rng_state())

def restore(state,controller,optimizer):
    controller.load_state_dict(state['controller']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    return state['cursor'],state['lineage_update'],Counter(state['coverage']),state['pruning_updates']

def save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial')
    torch.save(state,temp);temp.replace(path)
