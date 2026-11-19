"""Atomic on-demand cache of deterministic frozen encoder outputs, never noise/KV."""
import hashlib
import copy
import io
import json
import os
from pathlib import Path
import uuid
import torch
from bagel_robotics.data_selection.manifest import digest


def conditioning_identity(config):
    # These execution-only switches cannot change frozen encoder inputs/outputs.
    # Cache keys bind inputs, while equivalent execution switches share payloads.
    stable=copy.deepcopy(config)
    # Acceptance rules cannot change frozen encoder inputs or outputs.
    for key in ('cache_hard_masks','mixed_attention_checkpointing'):
        if key in stable.get('execution',{}):
            if not isinstance(stable['execution'][key],bool):raise ValueError('Invalid execution cache switch')
            del stable['execution'][key]
    return digest(stable)


class ConditioningCache:
    def __init__(self,config):
        self.identity=conditioning_identity(config)
        self.root=Path(config['runtime_conditioning_cache'])/self.identity
        self.root.mkdir(parents=True,exist_ok=True)
    def path(self,key):return self.root/f'{int(key):08d}.bin'
    def load(self,key):
        path=self.path(key)
        if not path.exists():return None
        with path.open('rb') as f:
            header=json.loads(f.readline());data=f.read()
        if header['identity']!=self.identity or header['window_id']!=int(key) or hashlib.sha256(data).hexdigest()!=header['sha256']:
            raise ValueError('Frozen conditioning cache identity/checksum mismatch')
        payload=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        if payload.get('input_latents') is not None:raise ValueError('Future video targets in conditioning cache')
        return payload
    def save(self,key,inputs):
        payload={k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in inputs.items()}
        payload['input_latents']=None
        buffer=io.BytesIO();torch.save(payload,buffer);data=buffer.getvalue()
        header=dict(identity=self.identity,window_id=int(key),sha256=hashlib.sha256(data).hexdigest())
        path=self.path(key);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
        with temp.open('wb') as f:f.write((json.dumps(header)+'\n').encode());f.write(data)
        try:
            # Atomically publish, without replacing another rank's complete entry.
            os.link(temp,path)
        except FileExistsError:pass
        finally:temp.unlink(missing_ok=True)
