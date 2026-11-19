"""Local metrics with optional experiment logging, isolated from training RNG."""
from functools import wraps
import json
from pathlib import Path
from bagel_robotics._internal.training.rng import diagnostic_rng
from bagel_robotics._internal.compact_kv.spec import atomic_json


def isolated_logging(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with diagnostic_rng():return function(*args,**kwargs)
    return wrapped


class Reporter:
    @isolated_logging
    def __init__(self,config,directory,stage,enabled=True):
        self.directory=Path(directory);self.run=None
        settings=config.get('wandb',{})
        if enabled and settings.get('enabled'):
            import wandb
            self.run=wandb.init(project=settings['project'],entity=settings.get('entity'),
                                name=settings.get('name'),group=settings.get('group'),config=config,
                                dir=str(directory))
    @isolated_logging
    def write(self,row):
        with (self.directory/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        if self.run:self.run.log(row,step=row['update'])
        if row['update']==1 or row['update']%10==0:
            print(json.dumps({key:row[key] for key in ('update','stage','gt_action','reliability','seconds_per_update') if key in row}),flush=True)
    @isolated_logging
    def native(self,result):
        if self.run:
            self.run.log({'native/'+key+'/'+name:value for key,record in result['actions'].items()
                          for name,value in record.items()},step=result['update'])
    @isolated_logging
    def masks(self,runtime,update,coverage,history):
        atomic_json(self.directory/f'diagnostics/selection_{update:06d}.json',dict(history,coverage=coverage))
    @isolated_logging
    def close(self):
        if self.run:self.run.finish()
