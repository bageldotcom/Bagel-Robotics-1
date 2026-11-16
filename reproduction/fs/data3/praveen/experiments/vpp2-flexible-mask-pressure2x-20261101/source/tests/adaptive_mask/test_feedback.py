import copy
import json
from pathlib import Path
import pytest
from vpp2.adaptive_mask.feedback import FittingFeedback, diagnostic_rows, selection_churn
from vpp2.adaptive_mask.budget import Budget
from vpp2.adaptive_mask.audit import build_manifest,validate_manifest
from vpp2.full_mask.panel import digest,BalancedStream,noise_stratum

PANEL=Path('/data3/praveen/experiments/vpp2-critical-calibration-20261029/revisions/3ca392bfb574898f/panel.json')


def examples():
    return [dict(id=t*100+e,task=str(t),suite='suite',episode=t*100+e,role='selection_initial')
            for t in range(2) for e in range(12)]


def observations(rows,ratio=1.,offset=0):
    return [dict(window_id=r['id'],position=offset+i*4+s,stratum=s,teacher_error=1.,student_error=ratio)
            for i,r in enumerate(rows) for s in range(4)]


def test_feedback_is_fitting_only_and_does_not_drop_rejected_teachers():
    rows=examples();f=FittingFeedback(rows);records=observations(rows,1.2)
    for r in records:r['reliability']=0
    f.add(records);result=f.report({})
    assert result['complete'] and result['resolved_violation'] and not result['passed']
    assert result['windows']==96 and not result['reliability_filtered']
    with pytest.raises(ValueError):f.add([dict(records[0],position=999,window_id=-1)])
    with pytest.raises(ValueError):FittingFeedback([dict(rows[0],role='monitor')])


def test_feedback_distributed_partition_and_resume_are_exact():
    rows=examples();records=observations(rows)
    a=FittingFeedback(rows);a.add(records)
    b=FittingFeedback(rows);b.add(sum([records[r::8] for r in range(8)],[]))
    c=FittingFeedback(rows);c.add(records[:37]);c=FittingFeedback(rows,json.loads(json.dumps(c.state_dict())));c.add(records[37:])
    assert a.state_dict()==b.state_dict()==c.state_dict()
    assert a.report({})['passed']
    with pytest.raises(ValueError):a.add(records[:1])


def test_sparse_or_conflicting_noise_stratum_holds_capacity():
    rows=examples();f=FittingFeedback(rows);f.add(observations(rows)[:8])
    assert not f.report({})['complete'] and not f.report({})['passed']
    f=FittingFeedback(rows);obs=observations(rows)
    for r in obs:
        if r['stratum']==3:r['student_error']=1.2
    f.add(obs);result=f.report({})
    assert result['resolved_violation'] and not result['passed']


def test_controller_holds_uncertainty_without_rollback_and_harm_gets_one_rescue():
    good=dict(source='fitting_only',complete=True,passed=True,resolved_violation=False)
    churn=dict(ffn_max=0.,cross_heads_max=0.)
    b=Budget(original=16_394_878_784)
    for _ in range(70):b.advance()
    b.observe_fitting(dict(good,passed=False),churn)
    for _ in range(70):b.advance()
    assert b.cap==b.original
    b.observe_fitting(good,churn)
    for _ in range(70):b.advance()
    assert b.cap<b.original
    b.observe_fitting(good,churn)
    for _ in range(70):b.advance()
    held=b.cap
    assert b.decide(False,resolved_violation=False)=='hold_quality_uncertain'
    assert not b.rescue_used and b.cap==held
    assert b.observe_fitting(good,churn)=='hold_selection_check'
    for _ in range(280):b.advance()
    assert b.cap==held
    assert b.decide(False,resolved_violation=True)=='rescue_confirmed_harm_once'
    for _ in range(560):b.advance()
    assert b.decide(False,resolved_violation=True)=='restore_parent_and_stop_shrinking'
    assert b.cap==b.original and b.stop_shrinking


def test_capacity_cannot_shrink_without_full_interval_or_when_churn_is_high():
    b=Budget(original=16_394_878_784)
    for _ in range(71):b.advance()
    assert b.cap==b.original
    with pytest.raises(ValueError):b.observe_fitting({},dict(ffn_max=0,cross_heads_max=0))
    for _ in range(69):b.advance()
    good=dict(source='fitting_only',complete=True,passed=True,resolved_violation=False)
    assert b.observe_fitting(good,dict(ffn_max=.06,cross_heads_max=0))=='hold_channel_turnover'
    # Final size is flexible; a larger accepted network can settle and finish.
    for _ in range(1820):b.advance()
    assert b.decide(True,stable=True,coverage=True)=='finish' and b.cap>b.target


def test_churn_does_not_confuse_capacity_reduction_with_identity_replacement():
    a=dict(ffn_indices=[[0,1,2,3]],cross_head_indices=[[0,1,2]])
    b=dict(ffn_indices=[[0,1]],cross_head_indices=[[0,1]])
    assert selection_churn(a,b)['ffn_max']==0
    b['ffn_indices']=[[4,5]]
    assert selection_churn(a,b)['ffn_max']==1


def test_real_panel_diagnostics_rotate_over_every_task_and_never_monitor():
    panel=json.loads(PANEL.read_text());seen=set()
    for n in range(12):
        rows=diagnostic_rows(panel,100008,n)
        assert len(rows)==5 and all(r['role']=='selection_initial' for r in rows)
        seen.update(r['task'] for r in rows)
    assert seen==set(panel['tasks'])


def test_real_first_feedback_interval_covers_every_task_noise_group():
    panel=json.loads(PANEL.read_text());rows=[r for r in panel['windows'] if r['role']=='selection_initial']
    stream=BalancedStream(rows,100008);f=FittingFeedback(rows)
    f.add([dict(window_id=stream.at(i)['id'],position=i,stratum=noise_stratum(100008,i,35),teacher_error=1.,student_error=1.)
           for i in range(70*256)])
    result=f.report({});assert result['passed'] and result['complete'] and result['windows']==70*256


def test_audit_manifest_is_predeclared_disjoint_complete_and_tamper_evident():
    root=Path('/data3/praveen/experiments/vpp2-adaptive-latekv-mask-search-20261030')
    config=json.loads((root/'config.json').read_text());panel=json.loads(PANEL.read_text())
    source=json.loads(Path(config['artifacts']['source_manifest']['path']).read_text())
    manifest=build_manifest(config,panel,source);assert validate_manifest(manifest,panel)
    assert manifest['windows']==1120 and manifest['episodes']==280
    assert all(r['frame']+32<r['length'] for r in manifest['rows'])
    assert len({r['id'] for r in manifest['rows']})==1120
    altered=copy.deepcopy(manifest);altered['rows'][0]['episode']=panel['windows'][0]['episode']
    altered['identity']=digest({k:v for k,v in altered.items() if k!='identity'})
    with pytest.raises(ValueError):validate_manifest(altered,panel)


def test_partial_feedback_resume_and_decision_match_uninterrupted():
    rows=examples();records=observations(rows)
    good=dict(source='fitting_only',complete=True,passed=True,resolved_violation=False)
    churn=dict(ffn_max=0.,cross_heads_max=0.)
    original=Budget(original=16_394_878_784);a=FittingFeedback(rows)
    for _ in range(70):original.advance()
    a.add(records[:37])
    resumed=Budget(**json.loads(json.dumps(original.state_dict())))
    b=FittingFeedback(rows,json.loads(json.dumps(a.state_dict())))
    # A stop just before the pending check must not consume another sample/update.
    for budget,feedback in ((original,a),(resumed,b)):
        feedback.add(records[37:])
        budget.observe_fitting(feedback.report({}),churn)
        feedback.clear()
        for _ in range(70):budget.advance()
    assert original.state_dict()==resumed.state_dict()
    assert a.state_dict()==b.state_dict() and original.progress==140


def test_maximum_budget_cannot_trigger_an_unfunded_rescue():
    b=Budget(original=16_394_878_784)
    for _ in range(b.maximum_updates):b.advance()
    assert b.decide(False,resolved_violation=True)=='stop_quality_failed'
    assert not b.rescue_used


def test_fitting_visualization_contains_task_noise_and_churn(tmp_path):
    from vpp2.adaptive_mask.report import plot_feedback
    f=FittingFeedback(examples());f.add(observations(examples()))
    b=Budget(original=16_394_878_784)
    path=plot_feedback(f.report({}),dict(ffn_by_layer=[0.,.1],cross_heads_by_layer=[.1,0]),b,70,tmp_path)
    assert path.stat().st_size>1000
