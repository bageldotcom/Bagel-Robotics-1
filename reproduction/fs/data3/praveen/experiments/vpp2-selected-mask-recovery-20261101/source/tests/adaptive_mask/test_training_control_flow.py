"""Execute the real trainer control loop with a tiny CPU policy oracle.

Only native model/data/GPU I/O are substituted. Optimizer, masks, exact parameter
allocator, sampling, feedback, hidden trials, checkpoints and finalization run.
This does NOT replace native GPU numerical verification.
"""
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from test_hidden import setup
from vpp2.adaptive_mask import train, preflight, data, report
from vpp2.critical_windows import inference


@pytest.mark.parametrize('action_only', [False, True])
@pytest.mark.parametrize('harmful_hidden', [False, True])
def test_real_training_loop_accepts_or_restores_hidden_trials(tmp_path, monkeypatch, harmful_hidden, action_only):
    c, inventory, _, _, _ = setup()
    with torch.no_grad():
        c.ffn_scores.copy_(torch.linspace(1,-1,128).expand_as(c.ffn_scores))
        c.cross_scores.copy_(torch.linspace(1,-1,4).expand_as(c.cross_scores))
    target=inventory.cost(80,[96,96,128],[3,3,4]);gap=inventory.original-target
    config=dict(root=str(tmp_path),seed=100008,
        panel_file=str(tmp_path/'panel.json'),hidden_search=dict(widths=[80,78,76],width=80),
        train=dict(global_batch=4,micro_batch=1,learning_rate=.01,betas=[.9,.999],warmup_updates=50,
                   information_updates=70,max_attempted_updates=2240,gradient_clip=1.,monitor_every=70,
                   max_full_checks_excluding_initial=8),
        budget=dict(target_parameters=target,milestones=[[560,inventory.original-gap//3],
            [1120,inventory.original-2*gap//3],[1680,target]],stability_updates=280,extra_updates=280,early_full_update=280),
        mask_priority={},capacity_feedback=dict(every=70,minimum_episodes_per_task_stratum=1,
            max_ffn_churn=.05,max_cross_head_churn=.25,max_hidden_churn=.005),quality={},
        wandb=dict(entity='test',project='test',group='test',name='test'),
        artifacts=dict(audit_manifest=dict(sha256='test-audit')))
    if action_only:config['quality_gate']={'mode':'action_only_v1'}
    panel=dict(identity='test-panel',tasks=['a','b'],windows=[dict(id=i,task='a' if i%2 else 'b',
        suite='memory' if i%2 else 'precision',episode=i,frame=0,role='selection_initial') for i in range(16)])
    for directory in ('checks','runs','control'):(tmp_path/directory).mkdir()
    (tmp_path/'panel.json').write_text(json.dumps(panel));(tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'checks/native.json').write_text(json.dumps(dict(checkpoint_hashes='fake-native')))
    monkeypatch.setattr(sys,'argv',['train','--config',str(tmp_path/'config.json')])
    monkeypatch.setattr(preflight,'validate_inputs',lambda *a:None)
    monkeypatch.setattr(train,'require_receipts',lambda *a:None)
    monkeypatch.setattr(train,'distributed',lambda:(0,1))
    monkeypatch.setattr(train,'gather',lambda x:[x]);monkeypatch.setattr(train,'barrier',lambda:None)
    monkeypatch.setattr(train.signal,'signal',lambda *a:None)
    monkeypatch.setattr(inference,'model_identity',lambda *a:'fake-native')
    model=torch.nn.Linear(1,1).requires_grad_(False)
    monkeypatch.setattr(train,'initialize',lambda *a:(model,None,c,None,inventory))
    monkeypatch.setattr(train,'prepare',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    original_zeros,original_as_tensor=torch.zeros,torch.as_tensor
    def cpu_only(function):
        def wrapped(*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return function(*args,**kwargs)
        return wrapped
    monkeypatch.setattr(torch,'zeros',cpu_only(original_zeros));monkeypatch.setattr(torch,'as_tensor',cpu_only(original_as_tensor))
    class Reporter:
        url='test://local'
        def define_metric(self,*a,**k):pass
        def log(self,*a,**k):pass
        def finish(self):pass
    monkeypatch.setitem(sys.modules,'wandb',types.SimpleNamespace(init=lambda **k:Reporter()))
    monkeypatch.setattr(report,'log_monitor',lambda *a:None);monkeypatch.setattr(report,'log_feedback',lambda *a:None)
    def harmful():return harmful_hidden and len(c.selection()['hidden_indices'])<80
    def quality(*args,**kwargs):
        return dict(passed=not harmful(),clearly_passing=not harmful(),resolved_violation=harmful())
    def monitor_quality(*args,**kwargs):
        if not action_only:return quality()
        from vpp2.adaptive_mask.quality_policy import monitor_acceptance
        action=quality();pruned=inventory.count(c.selection())<inventory.original
        video=dict(passed=not pruned,clearly_passing=not pruned,resolved_violation=pruned)
        return monitor_acceptance(config,action,video,{k:action for k in ('position','quaternion_coordinates','gripper')})
    monkeypatch.setattr(train,'evaluate',monitor_quality);monkeypatch.setattr(train,'native_evaluate',quality)
    monkeypatch.setattr(train,'gradient_diagnostics',lambda *a,**k:[dict(families={
        name:dict(active=True) for name in ('hidden','ffn','cross_heads')})])
    def step(*args):
        loss=sum(g.sum() for g in c.gates())*.0001
        return loss,{'loss/gt_action':loss.detach(),'loss/late_kv_applied':torch.tensor(0.)},dict(
            teacher_error=torch.tensor([1.]),student_error=torch.tensor([1.2 if harmful() else 1.])),{}
    monkeypatch.setattr(train,'step',step)
    train.main()
    final=json.loads((tmp_path/'final_selection.json').read_text())
    assert final['stable'] and final['attempt'] in (1960,2240)
    assert final['quality_gate_mode']==('action_only_v1' if action_only else 'action_and_video_v1')
    if action_only and not harmful_hidden:
        assert final['quality']['passed']  # Video diagnostic harm did not veto a good policy.
    assert len(final['selection']['hidden_indices'])==(80 if harmful_hidden else 76)
    assert final['hidden_search']['blocked']==harmful_hidden
    records=[json.loads(p.read_text()) for p in sorted((tmp_path/'runs').glob('monitor_attempt_*.json'))]
    decisions=[r['decision'] for r in records]
    assert 'begin_isolated_hidden_trial' in decisions
    if harmful_hidden:
        assert 'reject_hidden_confirmed_harm' in decisions
        rejected=next(r for r in records if r['decision']=='reject_hidden_confirmed_harm')
        assert len(rejected['record']['selection']['hidden_indices'])==78
        assert len(rejected['next_selection']['hidden_indices'])==80
    else:
        assert decisions.count('accept_hidden_width')==2
    saved=torch.load(tmp_path/'runs/latest.pt',weights_only=False)
    assert not saved['hidden_search']['pending']
    assert min(saved['state']['coverage'].values())>=2
    # Idempotent resume of a finished run must consume no new updates.
    train.main()
    assert json.loads((tmp_path/'final_selection.json').read_text())==final
