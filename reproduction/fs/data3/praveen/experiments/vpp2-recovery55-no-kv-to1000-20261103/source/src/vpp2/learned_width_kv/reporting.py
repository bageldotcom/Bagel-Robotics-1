"""One W&B identity for both phases; durable separate objective/mask diagnostics."""
import json
from pathlib import Path
from vpp2.compact_kv.reporting import Reporter,plot_log
from vpp2.compact_kv.spec import stable_seed
from .lifecycle import preserve_rng


class LearnedReporter(Reporter):
    @preserve_rng()
    def __init__(self,config,enabled=True):
        self.directory=Path(config['root'])/'metrics';self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'metrics.jsonl';self.config=config;self.candidate='learned-mask';self.kv_only=()
        self.wandb=None
        if enabled and config['wandb']['enabled']:
            import wandb
            wc=config['wandb']
            self.wandb=wandb.init(entity=wc['entity'],project=wc['project'],group=wc['group'],
                name='precision-six-tasks-learned-masks-1000-recovery-1000',
                id=f'learned-width-{stable_seed(config["seed"],config["root"]):x}',resume='allow',dir=str(self.directory),config=config)
            wandb.define_metric('update');wandb.define_metric('*',step_metric='update')
            (self.directory/'wandb_url.txt').write_text(self.wandb.url+'\n')

    @preserve_rng()
    def write(self,row):
        super().write(row)
        if self.wandb and 'by_scope' in row:
            values={'update':row['update']}
            for scope,metrics in row['by_scope'].items():
                for key,value in metrics.items():
                    if isinstance(value,(int,float)):
                        values[f"{row['kind']}/{scope}/{key}"]=value
                for i,(k,v) in enumerate(metrics.get('kv_by_layer_kv',[])):
                    values[f"{row['kind']}/{scope}/kv_layer/{i:02d}/k"]=k
                    values[f"{row['kind']}/{scope}/kv_layer/{i:02d}/v"]=v
                for i,(k,v) in enumerate(metrics.get('kv_gated_by_layer_kv') or []):
                    values[f"{row['kind']}/{scope}/kv_gated_layer/{i:02d}/k"]=k
                    values[f"{row['kind']}/{scope}/kv_gated_layer/{i:02d}/v"]=v
            self.wandb.log(values)
        if self.wandb and row.get('kv_gated_by_layer_kv') is not None:
            values={'update':row['update']}
            for i,(k,v) in enumerate(row['kv_gated_by_layer_kv']):
                values[f"{row['kind']}/kv_gated_layer/{i:02d}/k"]=k
                values[f"{row['kind']}/kv_gated_layer/{i:02d}/v"]=v
            self.wandb.log(values)
        if self.wandb and 'ffn_soft_count' in row:
            self.wandb.log({'update':row['update'],**{f'masks/ffn_{i:02d}_soft_count':v for i,v in enumerate(row['ffn_soft_count'])}})

    def plots(self):
        result=super().plots()
        if not self.path.exists(): return result
        rows=[json.loads(line) for line in self.path.read_text().splitlines()]
        rows=sorted({r['update']:r for r in rows if r.get('kind')=='train'}.values(),key=lambda r:r['update'])
        if not rows:return result
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fields=['action_teacher_weighted','teacher_action_weight','mask_count','mask_entropy','mask_discreteness',
                'mask_regularizer','hidden_soft_count','mask_gradient_norm','video_gradient_norm',
                'kv_gated','kv_effective_weight','teacher_supervision_rejected_fraction']
        fig,axes=plt.subplots(4,3,figsize=(16,14))
        for ax,key in zip(axes.flat,fields):
            data=[r for r in rows if key in r]
            ax.plot([r['update'] for r in data],[r[key] for r in data]);ax.set(title=key,xlabel='Global update');ax.grid(alpha=.2)
            ax.axvline(1000,color='gray',linestyle='--')
        fig.tight_layout();folder=self.directory/'plots';folder.mkdir(exist_ok=True)
        path=folder/'reliability_and_masks.png';fig.savefig(path,dpi=150);plt.close(fig)
        result.append(str(path))
        # Raw layer plots stay available through Reporter. Show gated errors on
        # a linear scale so a fully rejected update is visibly zero.
        import numpy as np
        gated=[r for r in rows if r.get('kv_gated_by_layer_kv') is not None]
        if gated:
            steps=np.array([r['update'] for r in gated])
            layers=np.array([r['kv_gated_by_layer_kv'] for r in gated])
            if layers.shape[1:]!=(40,2):raise ValueError('Invalid gated KV log shape')
            edges=np.r_[steps[0]-.5,(steps[1:]+steps[:-1])/2,steps[-1]+.5]
            fig,axes=plt.subplots(1,2,figsize=(17,10),sharey=True,layout='constrained')
            for j,ax in enumerate(axes):
                mesh=ax.pcolormesh(edges,np.arange(41)-.5,layers[:,:,j].T,
                    vmin=0.,vmax=max(float(layers.max()),1e-12),shading='flat',cmap='viridis')
                ax.set(title=f"{'K' if j==0 else 'V'}: reliability-weighted error",xlabel='Global update',ylabel='Layer index')
            fig.colorbar(mesh,ax=axes,label='Full-batch mean, before scheduled KV coefficient')
            path=folder/'kv_gated_layer_heatmaps.png';fig.savefig(path,dpi=150);plt.close(fig)
            result.append(str(path))
        return result
