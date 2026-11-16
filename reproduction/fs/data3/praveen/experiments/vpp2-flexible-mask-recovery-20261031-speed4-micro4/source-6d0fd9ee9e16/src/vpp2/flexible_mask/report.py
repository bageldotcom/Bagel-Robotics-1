"""Local visual logs plus the previously authorized W&B project."""
import json
from pathlib import Path
import numpy as np
from vpp2.compact_kv.spec import atomic_json
from .rng import diagnostic_rng
from .config import identity
from functools import wraps


def isolated_logging(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with diagnostic_rng():return function(*args,**kwargs)
    return wrapped


class Reporter:
    @isolated_logging
    def __init__(self,config,directory,stage,enabled=True):
        self.directory=Path(directory);self.run=None
        self.last_row=None;self.last_uploaded_update=None;self.last_printed_update=None
        self.plots=self.directory/'plots';self.plots.mkdir(exist_ok=True)
        if enabled:
            import wandb
            settings=config['wandb']
            self.run=wandb.init(entity=settings['entity'],project=settings['project'],group=settings['group'],
                name=settings['name']+'-'+stage,id='flexible-'+stage+'-'+identity(config)[:12],
                resume='allow',config=config,dir=str(directory))

    @isolated_logging
    def write(self,row):
        with (self.directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.last_row=dict(row)
        update=row['update'];calibrating=any(k.startswith('calibration/') for k in row)
        if self.run and (update==1 or update%5==0 or calibrating):
            self.run.log(row,step=update);self.last_uploaded_update=update
        if update==1 or update%10==0 or calibrating:
            self.progress(row)

    def progress(self,row):
        print(json.dumps({k:row[k] for k in ('update','stage','gt_action','late_kv_gated','reliability',
            'seconds_per_update','coefficient_measured','coefficient_capped') if k in row}),flush=True)
        self.last_printed_update=row['update']

    @isolated_logging
    def native(self,result):
        self.native_plot()
        if self.run:
            import wandb
            self.run.log({'native/'+key+'/'+name:value for key,record in result['actions'].items()
                          for name,value in record.items()},step=result['update'])
            self.run.log({'native/suites_plot':wandb.Image(str(self.plots/'native_suites.png')),
                          'native/tasks_plot':wandb.Image(str(self.plots/'native_tasks.png'))},step=result['update'])

    def native_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        records=[json.loads(p.read_text()) for p in sorted((self.directory/'diagnostics').glob('native_*.json'))]
        if not records:return
        suites=sorted(k for k in records[-1]['actions'] if k.startswith('suite/'))
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for suite in suites:
            axes[0].plot([r['update'] for r in records],[r['actions'][suite]['student_error'] for r in records],label=suite[6:])
            axes[1].plot([r['update'] for r in records],[(r['actions'][suite]['student_error']/max(r['actions'][suite]['teacher_error'],1e-8)-1)*100 for r in records],label=suite[6:])
        axes[0].set(xlabel='Update',ylabel='Generated-action MSE',title='Offline native ten-step inference; first 24 actions')
        axes[0].set_yscale('symlog',linthresh=1e-8);axes[0].legend()
        axes[1].axhline(0,color='black',linewidth=.8);axes[1].set(xlabel='Update',ylabel='Change from teacher (%)',title='Diagnostic comparison; no teacher-closeness gate')
        axes[1].legend();fig.tight_layout();fig.savefig(self.plots/'native_suites.png',dpi=130);plt.close(fig)
        tasks=sorted(k for k in records[-1]['actions'] if k.startswith('task/'))
        values=np.array([[(r['actions'][task]['student_error']/max(r['actions'][task]['teacher_error'],1e-8)-1)*100 for r in records] for task in tasks])
        fig,ax=plt.subplots(figsize=(12,12));im=ax.imshow(values,aspect='auto',cmap='RdBu_r')
        ax.set_yticks(range(len(tasks)),[x[5:] for x in tasks]);ax.set_xticks(range(len(records)),[r['update'] for r in records],rotation=90)
        ax.set(xlabel='Update',title='Task-wise generated-action MSE change from teacher (%)')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(self.plots/'native_tasks.png',dpi=130);plt.close(fig)

    @isolated_logging
    def masks(self,runtime,update,coverage,history):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selection=history['selection']
        probability=[p.detach().float().cpu().numpy() for p in runtime.masks.probabilities(update)]
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        axes[0,0].bar(range(40),list(map(len,selection['ffn_indices'])))
        axes[0,0].set(xlabel='Video block',ylabel='Retained FFN channels')
        axes[0,1].bar(range(40),list(map(len,selection['cross_head_indices'])))
        axes[0,1].set(xlabel='Video block',ylabel='Retained conditioning heads')
        axes[1,0].hist(probability[0],bins=40,range=(0,1));axes[1,0].set(xlabel='Hidden mask value',ylabel='Count')
        axes[1,1].imshow(probability[2],aspect='auto',vmin=0,vmax=1);axes[1,1].set(xlabel='Conditioning head',ylabel='Video block')
        fig.suptitle(f'Update {update}: physical video size {history["parameters"]/1e9:.3f}B; soft masks are not a compact model')
        fig.tight_layout();path=self.plots/f'masks_{update:06d}.png';fig.savefig(path,dpi=130);plt.close(fig)
        ffn=probability[1].reshape(39,-1,128).mean(-1)
        fig,ax=plt.subplots(figsize=(12,6));im=ax.imshow(ffn,aspect='auto',vmin=0,vmax=1)
        ax.set(xlabel='FFN channel group (128 original coordinates)',ylabel='Video block',title='Mean mask value; individual channel identities remain separate')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(self.plots/f'ffn_{update:06d}.png',dpi=130);plt.close(fig)
        lines=[json.loads(x) for x in (self.directory/'metrics.jsonl').read_text().splitlines()]
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for name in ('gt_action','teacher_video_gated','late_kv_gated'):
            axes[0,0].plot([r['update'] for r in lines],[r[name] for r in lines],label=name)
        axes[0,0].legend();axes[0,0].set_yscale('symlog',linthresh=1e-8)
        for name in ('size_weight','coefficient_measured','coefficient_capped'):
            axes[0,1].plot([r['update'] for r in lines],[r.get(name) for r in lines],label=name)
        axes[0,1].legend();axes[0,1].set_title('Measured coefficient, cap and applied ramp')
        axes[1,0].plot([r['update'] for r in lines],[r['reliability'] for r in lines]);axes[1,0].set_title('Teacher reliability; full batch denominator')
        kv=np.array([[r[f'kv/layer_{i:02d}_k'] for i in range(40)] for r in lines]).T
        axes[1,1].imshow(np.log10(np.maximum(kv,1e-12)),aspect='auto',origin='lower')
        axes[1,1].set(xlabel='Logged update',ylabel='Video block',title='log10 K error (0–19 diagnostic only)')
        fig.tight_layout();fig.savefig(self.plots/'objectives.png',dpi=130);plt.close(fig)
        saturation={name:dict(below_point_one=float((p<.1).mean()),above_point_nine=float((p>.9).mean()),
            near_boundary=float(((p>=.4)&(p<=.6)).mean())) for name,p in zip(('hidden','ffn','cross'),probability)}
        atomic_json(self.directory/f'diagnostics/selection_{update:06d}.json',dict(history,coverage=coverage,saturation=saturation))
        if self.run:
            import wandb
            self.run.log({'masks/selection':wandb.Image(str(path)),
                'masks/ffn_values':wandb.Image(str(self.plots/f'ffn_{update:06d}.png')),
                'objectives/overview':wandb.Image(str(self.plots/'objectives.png')),
                'parameters/physical':history['parameters'],
                'parameters/soft':float(runtime.cost.parameters(runtime.masks.gates(update))),
                **{'coverage/'+task+'/cycles':v['cycles'] for task,v in coverage.items()}},step=update)

    @isolated_logging
    def close(self):
        # Always publish the final completed update, including an interruption
        # between regular logging intervals. Full local records are never thinned.
        if self.last_row is not None:
            update=self.last_row['update']
            if self.run and self.last_uploaded_update!=update:
                self.run.log(self.last_row,step=update);self.last_uploaded_update=update
            if self.last_printed_update!=update:self.progress(self.last_row)
        if self.run:self.run.finish()
