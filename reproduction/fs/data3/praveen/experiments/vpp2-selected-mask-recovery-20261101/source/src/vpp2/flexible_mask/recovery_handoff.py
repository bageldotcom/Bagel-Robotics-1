"""Explicit early-recovery handoff, distinct from mask-convergence certification."""
import json
from pathlib import Path
from collections import OrderedDict
import copy
import torch
from .config import identity
from .export import read_export_manifest
from vpp2.full_mask.panel import file_hash,digest


def require_selected(config,export):
    root=Path(config['root']);entry=config.get('recovery_handoff',{})
    if entry.get('schema')!='milestone-selection-recovery-v1' or entry.get('early_recovery_authorized') is not True:
        raise ValueError('Missing explicit early-recovery authorization')
    if config.get('resume_from') or config.get('pressure_continuation'):
        raise ValueError('Mask optimizer/cursor/pressure cannot continue into recovery')
    selected=json.loads((root/'selected.json').read_text())
    if selected['identity']!=identity(config) or selected['status']!='selected_for_early_recovery':
        raise ValueError('Wrong recovery selection identity')
    report=Path(selected['selection_report'])
    if file_hash(report)!=selected['selection_report_sha256']:raise ValueError('Selection report changed')
    result=json.loads(report.read_text())
    if not result['coverage_complete'] or result['protocol_identity']!=entry['protocol_identity']:
        raise ValueError('Incomplete or different selection comparison')
    selection_root=Path(entry['selection_root'])
    if file_hash(selection_root/'selection_rule.json')!=entry['selection_rule_sha256']:
        raise ValueError('Selection rule changed')
    if result['ranking_rule']!=json.loads((selection_root/'selection_rule.json').read_text()):
        raise ValueError('Different ranking rule used')
    m=read_export_manifest(export,require_verified=True)
    c=result['recommended_checkpoint']
    if (m['identity']!=identity(config) or m['checkpoint_identity']!=c['checkpoint_identity']
        or m['mask_identity']!=c['mask_identity'] or m['update']!=result['recommended_update']
        or m['parameters']!=c['parameters'] or digest(m['selection'])!=c['mask_identity']):
        raise ValueError('Recovery export is not the exact selected checkpoint/mask')
    return m


class RecoveryMemoryCache:
    """Bounded lossless frozen-conditioning cache; no persistent window expansion."""
    def __init__(self,capacity=128):
        if type(capacity) is not int or capacity<8:raise ValueError('Cache must fit a complete microbatch')
        self.capacity=capacity;self.entries=OrderedDict()
    def load(self,key):
        if key not in self.entries:return None
        self.entries.move_to_end(key)
        return {k:(v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in self.entries[key].items()}
    def save(self,key,inputs):
        payload={k:(v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in inputs.items()}
        payload['input_latents']=None
        self.entries[key]=payload;self.entries.move_to_end(key)
        while len(self.entries)>self.capacity:self.entries.popitem(last=False)
