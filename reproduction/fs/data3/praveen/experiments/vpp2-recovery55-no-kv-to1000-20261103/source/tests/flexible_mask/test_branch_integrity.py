import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from vpp2.flexible_mask import checkpoint,low_lr,recovery_subset,recovery_best
from vpp2.flexible_mask.preflight import fingerprint
from vpp2.flexible_mask.rng import rng_state
from vpp2.flexible_mask.recovery_gradients import summarize,engine_counters,restore_counters,adam_state
from vpp2.flexible_mask.selection import aggregate


def evidence(path,update,value):
    rows=[dict(window_id=4*t+i,episode=4*t+i,frame=0,executed_steps=24,task=f't{t}',suite=f's{t%5}',
               student_error=value,teacher_error=.1) for t in range(35) for i in range(4)]
    path.write_text(json.dumps(dict(update=update,full=False,simulation=False,native_10_step_sampler=True,
                                    rows=rows,actions=aggregate(rows))))


def test_gradient_cosines_use_coordinate_sums():
    values=dict(gt={'video.blocks.20.x':torch.tensor([1.,0.])},
                video={'video.blocks.20.x':torch.tensor([0.,2.])},
                kv={'video.blocks.20.x':torch.tensor([-3.,0.])})
    with patch('vpp2.flexible_mask.recovery_gradients.dist.all_reduce',lambda t,op:None):r=summarize(values,'cpu')
    assert r['all']['gt_norm']==1 and r['all']['kv_norm']==3
    assert r['all']['gt_kv_cosine']==-1 and r['all']['gt_video_cosine']==0
    assert r['all']==r['block_20']


def test_engine_counter_snapshot_copies_mutable_values():
    e=SimpleNamespace(global_steps=55,micro_steps=440,losses=[1.,2.],total_loss=torch.ones(1),
                      optimizer=SimpleNamespace(),is_gradient_accumulation_boundary=lambda:False)
    saved=engine_counters(e);e.losses.append(3);e.total_loss.add_(1);restore_counters(e,saved)
    assert e.losses==[1.,2.] and torch.equal(e.total_loss,torch.ones(1))


def test_adam_snapshot_handles_zero_empty_parameter_groups():
    p=torch.nn.Parameter(torch.ones(3));opt=torch.optim.AdamW([p],lr=1e-6)
    p.sum().backward();opt.step();opt.zero_grad();opt.param_groups[0]['params']=[]
    e=SimpleNamespace(optimizer=SimpleNamespace(optimizer=opt,fp32_partitioned_groups_flat=[p]))
    s=adam_state(e)
    assert len(s['states'])==1 and 'exp_avg' in s['states'][0]
    assert opt.param_groups[0]['params']==[]


def test_restore55_retains_adam_rng_and_starts_original_curated_sequence(tmp_path,monkeypatch):
    monkeypatch.setattr(checkpoint.dist,'barrier',lambda:None)
    monkeypatch.setattr(checkpoint.dist,'all_gather_object',lambda out,v:out.__setitem__(0,v))
    monkeypatch.setattr('vpp2.flexible_mask.config.identity',lambda c:'new')
    parent=tmp_path/'original';child=tmp_path/'child'
    for r in [parent,child]:(r/'recovery/diagnostics').mkdir(parents=True)
    baseline=parent/'recovery/diagnostics/native_000055.json';evidence(baseline,55,1.)
    class Engine:
        def __init__(self):
            self.module=torch.nn.Linear(2,2);self.optimizer=torch.optim.AdamW(self.module.parameters(),lr=5e-6)
            self.module(torch.ones(1,2)).square().mean().backward();self.optimizer.step();self.optimizer.zero_grad()
            self.global_steps=55
        def save_checkpoint(self,root,tag,client_state,save_latest):
            p=Path(root)/tag;p.mkdir(parents=True)
            torch.save(dict(model=self.module.state_dict(),adam=self.optimizer.state_dict(),client=client_state,
                            steps=self.global_steps),p/'rank.pt')
        def load_checkpoint(self,root,tag,**kwargs):
            d=torch.load(Path(root)/tag/'rank.pt',weights_only=False)
            self.module.load_state_dict(d['model']);self.optimizer.load_state_dict(d['adam']);self.global_steps=d['steps']
            return str(Path(root)/tag),d['client']
    rt=SimpleNamespace(rank=0,world=1,engine=Engine(),masks=None,mask_optimizer=None,
                       recovery_export_sha256='export',physical_parameters=6)
    state=dict(identity='old55',update=55,stage='recovery',best=[],history=[],coefficient={},
               previous_scores=None,interval_grad=torch.zeros(3),interval_n=0)
    checkpoint.save(rt,parent/'recovery',state)
    c=dict(root=str(child),train={'global_batch':256},recovery={'updates':2344},
        recovery_subset=dict(parent_root=str(parent),parent_update=55,parent_checkpoint_identity='subset',
                             parent_identity='old55',parent_tag='update_000055'),
        recovery_lr_branch=dict(parent_root=str(parent),parent_identity='old55',parent_tag='update_000055',
                                schedule_endpoint=2399,baseline55_validation=str(baseline)))
    receipt=json.loads((parent/'recovery/checkpoints/update_000055/COMPLETE.json').read_text())
    monkeypatch.setattr(low_lr,'verify_parent',lambda c:(receipt,{'historical_checkpoint250_available':False}))
    monkeypatch.setattr(recovery_subset,'validate_parent',lambda c:({'root':str(parent)},receipt))
    before=fingerprint(dict(weights=rt.engine.module.state_dict(),adam=rt.engine.optimizer.state_dict(),rng=rng_state()))
    with torch.no_grad():
        for p in rt.engine.module.parameters():p.add_(10)
    torch.manual_seed(123)
    loaded=low_lr.load_parent(rt,c)
    after=fingerprint(dict(weights=rt.engine.module.state_dict(),adam=rt.engine.optimizer.state_dict(),rng=rng_state()))
    assert before==after
    assert loaded['update']==55 and loaded['cursor']==14080 and loaded['recovery_subset_cursor']==0
    assert [v['update'] for v in loaded['best']]==[55]
    for u in [56,100,250]:assert recovery_subset.sample_position(c,(u-1)*256)==(u-56)*256
    assert (parent/'recovery/checkpoints/update_000055/COMPLETE.json').exists()


def test_external_artifacts_never_enter_child_cleanup(tmp_path,monkeypatch):
    parent=tmp_path/'parent/recovery/best_models/update_000250';parent.mkdir(parents=True)
    (parent/'COMPLETE.json').write_text('{"identity":"parent","update":250,"weights_only":true}')
    child=tmp_path/'child';(child/'recovery/best_models').mkdir(parents=True)
    monkeypatch.setattr('vpp2.flexible_mask.config.identity',lambda c:'child')
    recovery_best.prune_exports({'root':str(child)},[],250);checkpoint.retain(child/'recovery',[])
    assert (parent/'COMPLETE.json').exists()


def test_training_and_validation_logs_share_update_axis(tmp_path):
    from vpp2.flexible_mask.report import Reporter
    r=Reporter({'recovery_lr_branch':{}},tmp_path,'recovery',enabled=False);r.use_update_axis=True
    class Fake:
        def __init__(self):self.calls=[]
        def log(self,row,**kwargs):self.calls.append((row,kwargs))
    r.run=Fake();r._log({'training':1},step=150);r._log({'validation':2},step=150)
    assert len(r.run.calls)==2 and all(row['update']==150 and kw=={} for row,kw in r.run.calls)
