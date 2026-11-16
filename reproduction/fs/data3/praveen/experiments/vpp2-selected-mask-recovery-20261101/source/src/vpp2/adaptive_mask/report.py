"""Local scientific plots plus W&B; plotting runs only at monitoring boundaries."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_monitor(monitor,native,diagnostics,selection,attempt,root):
    root=Path(root)/'figures';root.mkdir(parents=True,exist_ok=True);files=[]
    def finish(fig,name):
        path=root/f'{name}_{attempt:06d}.png';fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig);files.append(path)
    rows=monitor['rows'];kv=np.mean([r['kv'] for r in rows],axis=0)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for j,key in enumerate(('K','V')):
        axes[j].bar(np.arange(40),kv[:,j],color=['#88939b']*20+['#008b84']*20)
        axes[j].axvline(19.5,color='black',linestyle='--');axes[j].set_title(key+' error: gray diagnostic only; teal supervised')
        axes[j].set_xlabel('Zero-based layer');axes[j].set_ylabel('Teacher-energy-normalized MSE')
    finish(fig,'layer_kv')
    fig,axes=plt.subplots(2,2,figsize=(12,8));tasks=sorted({r['task'] for r in rows})
    axes[0,0].hist([r['base_reliability'] for r in rows],bins=np.linspace(0,1,21),alpha=.6,label='Before veto')
    axes[0,0].hist([r['reliability'] for r in rows],bins=np.linspace(0,1,21),alpha=.6,label='Applied');axes[0,0].legend();axes[0,0].set_title('Reliability weight distribution')
    values=[sum(r[k] for r in rows)/len(rows) for k in ('student_veto','zero_veto','component_veto')]
    axes[0,1].bar(['Student better','Zero-flow better','Component error'],values);axes[0,1].set_title('Veto rates (may overlap)')
    axes[1,0].bar(range(40),[len(s) for s in selection['ffn_indices']]);axes[1,0].set_title('Retained FFN channels by layer')
    axes[1,1].bar(range(40),[len(s) for s in selection['cross_head_indices']]);axes[1,1].set_title('Retained cross-attention heads by layer')
    finish(fig,'reliability_capacity')
    fig,axes=plt.subplots(2,1,figsize=(16,8))
    for ax,name,quality in zip(axes,['Action flow','Demonstrated-future video flow'],[monitor['action_quality'],monitor['video_quality']]):
        q=quality['original']['by_task'];labels=sorted(q);x=np.arange(len(labels))
        delta=np.array([q[t]['delta'] for t in labels]);lower=np.array([q[t]['lower'] if q[t]['lower'] is not None else np.nan for t in labels]);upper=np.array([q[t]['upper'] if q[t]['upper'] is not None else np.nan for t in labels]);allow=np.array([q[t]['allowance'] for t in labels])
        ax.errorbar(x,delta,yerr=np.stack([delta-lower,upper-delta]),fmt='o',label='Paired delta with adjusted bounds')
        ax.plot(x,allow,'r_',label='Allowed regression');ax.axhline(0,color='gray');ax.set_xticks(x,labels,rotation=70,fontsize=7);ax.set_title(name);ax.legend(fontsize=8)
    finish(fig,'paired_task_quality')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,records,name in [(axes[0],monitor['video_rows'],'Video GT error'),(axes[1],[] if native is None else native['rows'],'Native 10-step action GT error')]:
        lookup={r['window_id']:r for r in rows};bins=[[],[],[],[]]
        for row in records:
            if row['window_id'] in lookup:
                p=lookup[row['window_id']]['base_reliability'];idx=min(int(p*4),3);bins[idx].append(row)
        means=[np.mean([r['student_error'] for r in b]) if b else np.nan for b in bins]
        ref=[np.mean([r['teacher_error'] for r in b]) if b else np.nan for b in bins]
        ax.plot(range(4),means,'o-',label='Masked');ax.plot(range(4),ref,'o-',label='Original');ax.set_xticks(range(4),['0–.25','.25–.5','.5–.75','.75–1']);ax.set_title(name);ax.set_xlabel('Base reliability bin');ax.legend()
    finish(fig,'reliability_proxy_check')
    if diagnostics:
        fig,ax=plt.subplots(figsize=(9,4));families=sorted(diagnostics[0]['families']);matrix=np.array([[d['families'][f]['retained_gt_sensitivity'] for f in families] for d in diagnostics])
        im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1);ax.set_yticks(range(len(diagnostics)),[d['suite'] for d in diagnostics]);ax.set_xticks(range(len(families)),families);ax.set_title('Retained action-loss gradient mass (diagnostic, not accuracy)');fig.colorbar(im,ax=ax)
        finish(fig,'suite_sensitivity')
    if native and native['examples']:
        example=native['examples'][0];fig,axes=plt.subplots(4,4,figsize=(14,8))
        for d,ax in enumerate(axes.flat):
            for key in ('target','teacher','student'):ax.plot(np.array(example[key])[:,d],label=key)
            ax.axvline(23.5,color='gray',linestyle='--');ax.set_title('Action dimension '+str(d),fontsize=8)
        axes.flat[0].legend(fontsize=7);fig.suptitle(example['task']+' — native sampled actions; first 24 executed')
        finish(fig,'native_action_trajectories')
    if monitor['examples']:
        ex=monitor['examples'][0];fig,axes=plt.subplots(2,2,figsize=(10,6))
        for i,key in enumerate(('teacher_future','student_future')):
            for j in range(2):
                axes[i,j].imshow(np.array(ex[key])[0,j]);axes[i,j].set_title(key+' plane '+str(j+1));axes[i,j].axis('off')
        fig.suptitle(ex['label']);finish(fig,'future_latent_examples')
    # Cross-update traces use durable records, including rejected attempts.
    history=[]
    for path in sorted((Path(root).parent/'runs').glob('monitor_attempt_*.json')):
        value=json.loads(path.read_text());history.append(value)
    if history:
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for j,key in enumerate(('k','v')):
            mat=np.stack([np.mean([r['kv'] for r in h['monitor']['rows']],axis=0)[:,j] for h in history],axis=1)
            im=axes[j].imshow(np.log10(np.maximum(mat,1e-12)),aspect='auto',origin='lower');axes[j].axhline(19.5,color='white');axes[j].set_title(key.upper()+' error log10');axes[j].set_xlabel('Monitoring attempt');axes[j].set_ylabel('Layer');fig.colorbar(im,ax=axes[j])
        finish(fig,'kv_history')
        from .quality import overlap
        updates=[h['record']['attempt'] for h in history]
        fig,axes=plt.subplots(2,2,figsize=(14,8))
        axes[0,0].plot(updates,[len(h['record']['selection']['hidden_indices']) for h in history],'o-');axes[0,0].set_title('Retained hidden coordinates')
        for j,family in enumerate(('ffn_indices','cross_head_indices')):
            matrix=np.stack([[len(x) for x in h['record']['selection'][family]] for h in history],axis=1)
            im=axes[0,1] if j==0 else axes[1,0]
            obj=im.imshow(matrix,aspect='auto',origin='lower');im.set_title('FFN widths' if j==0 else 'Cross-attention head counts');fig.colorbar(obj,ax=im)
        for h in history:
            ratio=np.mean([v['ratio'] for v in h['monitor']['action_quality']['original']['by_task'].values()])
            color='green' if h['decision']=='accept' else 'red' if h['decision'].startswith('restore') else 'gray'
            axes[1,1].scatter(h['record']['cap']/1e9,ratio,c=color)
        axes[1,1].set_xlabel('Video parameter allowance (B)');axes[1,1].set_ylabel('Mean task action-error ratio');axes[1,1].set_title('Accepted (green), rejected (red), ongoing (gray)')
        finish(fig,'budget_quality')
        if len(history)>1:
            changes=[]
            for prev,now in zip(history,history[1:]):
                a,b=prev['record']['selection'],now['record']['selection']
                changes.append([len(set(x)^set(y))/max(len(set(x)|set(y)),1) for x,y in zip(a['ffn_indices'],b['ffn_indices'])])
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            im=axes[0].imshow(np.array(changes).T,aspect='auto',origin='lower',vmin=0,vmax=1);axes[0].set_title('FFN membership changes by layer');fig.colorbar(im,ax=axes[0])
            for family in ('hidden_indices','ffn_indices','cross_head_indices'):
                rates=[]
                for prev,now in zip(history,history[1:]):
                    a,b=prev['record']['selection'][family],now['record']['selection'][family]
                    if family=='hidden_indices':a=[a];b=[b]
                    rates.append(sum(len(set(y)-set(x)) for x,y in zip(a,b)))
                axes[1].plot(updates[1:],rates,'o-',label=family)
            axes[1].legend();axes[1].set_title('Coordinates/heads returning or entering the selection')
            finish(fig,'mask_membership')
    training_path=Path(root).parent/'runs/training.jsonl'
    if training_path.exists():
        logs=[json.loads(line) for line in training_path.read_text().splitlines() if line.strip()]
        if logs:
            fig,axes=plt.subplots(2,2,figsize=(14,8));x=[v['attempt'] for v in logs]
            for key in ('gt_action','late_kv_raw','video_raw'):
                axes[0,0].plot(x,[v['loss/'+key] for v in logs],label=key)
            for key in ('gt_action','late_kv_applied','video_applied'):
                axes[0,1].plot(x,[v['loss/'+key] for v in logs],label=key)
            axes[1,0].plot(x,[v['reliability/base'] for v in logs],label='Base reliability');axes[1,0].plot(x,[v['reliability/applied'] for v in logs],label='After veto')
            axes[1,1].plot(x,[v['unique_windows_lineage'] for v in logs],label='Accepted lineage unique');axes[1,1].plot(x,[v['unique_windows_attempted'] for v in logs],label='All attempted unique')
            for ax,title in zip(axes.flat,['Raw losses','Weighted objective contributions','Reliability','Actual coverage']):ax.set_title(title);ax.set_xlabel('Attempted optimizer update');ax.legend(fontsize=8)
            finish(fig,'training_objectives')
            if all('priority/ffn/uncertain_fraction' in v for v in logs):
                families=['ffn','cross_heads']+(['hidden'] if all('priority/hidden/uncertain_fraction' in v for v in logs) else [])
                fig,axes=plt.subplots(1,len(families),figsize=(6*len(families),4))
                for family,ax in zip(families,axes):
                    for part in ('uncertain_fraction','stable_keep_fraction','stable_off_fraction'):
                        ax.plot(x,[v[f'priority/{family}/{part}'] for v in logs],label=part)
                    ax.set_title(family+' mask-update priority');ax.set_ylim(0,1);ax.set_xlabel('Optimizer update');ax.legend(fontsize=7)
                finish(fig,'mask_priority')
    if monitor.get('component_quality'):
        fig,axes=plt.subplots(1,2,figsize=(16,7))
        for ax,record,label in [(axes[0],monitor,'Action flow'),(axes[1],native,'Native 10-step actions')]:
            if not record or not record.get('component_quality'):
                ax.set_axis_off();continue
            names=list(record['component_quality']);tasks=sorted(record['component_quality'][names[0]]['original']['by_task'])
            matrix=np.array([[record['component_quality'][name]['original']['by_task'][t]['ratio'] for name in names] for t in tasks])
            im=ax.imshow(matrix,aspect='auto',vmin=.95,vmax=1.10,cmap='RdYlGn_r')
            ax.set_yticks(range(len(tasks)),tasks,fontsize=6);ax.set_xticks(range(len(names)),names,rotation=20,fontsize=8)
            ax.set_title(label+' error / original, by component');fig.colorbar(im,ax=ax)
        finish(fig,'action_component_quality')
    if history:
        width=max(max(h['record']['selection']['hidden_indices']) for h in history)+1
        matrix=np.zeros((width,len(history)),dtype=np.uint8)
        for j,h in enumerate(history):matrix[h['record']['selection']['hidden_indices'],j]=1
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        axes[0].imshow(matrix,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='Greys')
        axes[0].set_title('Hidden coordinates: retained black / removed white');axes[0].set_ylabel('Original channel index')
        axes[0].set_xlabel('Monitoring snapshot')
        attempts=[h['record']['attempt'] for h in history]
        axes[1].step(attempts,[len(h['record']['selection']['hidden_indices']) for h in history],where='post')
        for h in history:
            if 'hidden' in h['decision']:axes[1].axvline(h['record']['attempt'],alpha=.4)
        axes[1].set_xlabel('Attempted update');axes[1].set_ylabel('Retained hidden width');axes[1].set_title('Hidden-width trials and retained width')
        finish(fig,'hidden_selection_history')
    return files

def log_monitor(reporter,monitor,native,diagnostics,selection,attempt,root):
    import wandb
    logs={'monitor/acceptance_passed':monitor['passed'],
          'monitor/video_is_acceptance_gate':int(monitor.get('video_is_acceptance_gate',True)),
          'monitor/video_diagnostic_passed':monitor['video_quality']['passed'],
          'monitor/video_diagnostic_harm':monitor['video_quality']['resolved_violation'],
          'monitor/quality_gate_mode':monitor.get('quality_gate_mode','action_and_video_v1')}
    qualities=[('action',monitor['action_quality']),('video',monitor['video_quality'])]
    qualities += [('action_'+k,v) for k,v in monitor.get('component_quality',{}).items()]
    if native:
        qualities += [('native_action',native['quality'])]
        qualities += [('native_'+k,v) for k,v in native.get('component_quality',{}).items()]
    for label,q in qualities:
        for task,r in q['original']['by_task'].items():
            for key in ('student','reference','ratio','delta','upper','allowance'):logs[f'monitor/{label}/{task}/{key}']=r[key]
        for suite,r in q['original']['by_suite'].items():
            for key in ('student','reference','delta','upper','allowance'):logs[f'suite/{label}/{suite}/{key}']=r[key]
    for d in diagnostics:
        for family,values in d['families'].items():
            for k,v in values.items():logs[f'gradient/{d["suite"]}/{family}/{k}']=v
    for path in plot_monitor(monitor,native,diagnostics,selection,attempt,root):logs['figures/'+path.stem.rsplit('_',1)[0]]=wandb.Image(str(path))
    logs['attempt']=attempt;reporter.log(logs)


def plot_feedback(report,churn,budget,attempt,root):
    directory=Path(root)/'figures';directory.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    tasks=sorted({t for value in report['strata'].values() for t in value['quality'].get('by_task',{})})
    matrix=np.full((len(tasks),4),np.nan)
    for s,value in report['strata'].items():
        for i,t in enumerate(tasks):
            result=value['quality'].get('by_task',{}).get(t)
            if result:matrix[i,int(s)]=result['ratio']
    if tasks:
        im=axes[0].imshow(matrix,aspect='auto',vmin=.95,vmax=1.10,cmap='RdYlGn_r')
        axes[0].set_yticks(range(len(tasks)),tasks,fontsize=6);axes[0].set_xticks(range(4),range(4))
        fig.colorbar(im,ax=axes[0],label='Masked / original GT action error')
    axes[0].set_title('Fitting-only task / action-noise feedback');axes[0].set_xlabel('Noise stratum')
    axes[1].plot(churn['ffn_by_layer'],label='FFN identity replacement')
    axes[1].plot(churn['cross_heads_by_layer'],label='Cross-attention head replacement')
    axes[1].set_ylim(0,1);axes[1].set_xlabel('Layer');axes[1].legend()
    axes[1].set_title(f'{budget.cap/1e9:.3f}B allowance; {budget.last_feedback}')
    fig.tight_layout();path=directory/f'fitting_controller_{attempt:06d}.png';fig.savefig(path,dpi=130);plt.close(fig)
    return path


def log_feedback(reporter,report,churn,budget,selection_decision,attempt,root):
    import wandb
    logs={'controller/decision':budget.last_feedback,'controller/selection_decision':selection_decision,
          'controller/authorized_updates':budget.authorized_updates,'controller/cap_B':budget.cap/1e9,
          'controller/fitting_complete':report['complete'],'controller/fitting_passed':report['passed'],
          'controller/fitting_harm':report['resolved_violation'],'controller/ffn_churn':churn['ffn_max'],
          'controller/head_churn':churn['cross_heads_max'],'controller/hidden_churn':churn.get('hidden',0.)}
    for s,value in report['strata'].items():
        for grouping in ('by_task','by_suite'):
            for label,metrics in value['quality'].get(grouping,{}).items():
                for k in ('reference','student','delta','upper','allowance','passed','resolved_violation'):
                    if k in metrics:logs[f'fitting/{grouping}/{label}/noise_{s}/{k}']=metrics[k]
    logs['figures/fitting_controller']=wandb.Image(str(plot_feedback(report,churn,budget,attempt,root)))
    logs['attempt']=attempt;reporter.log(logs)
