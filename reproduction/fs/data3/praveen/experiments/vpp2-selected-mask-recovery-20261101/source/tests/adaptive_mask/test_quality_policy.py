import copy
import pytest
from vpp2.adaptive_mask.quality_policy import ACTION_ONLY, LEGACY, policy_mode, monitor_acceptance
from vpp2.adaptive_mask.monitor import quality_settings
from vpp2.adaptive_mask.quality import stable
from vpp2.adaptive_mask.train import combined_quality
from vpp2.adaptive_mask.budget import Budget
from vpp2.full_mask.conditioning_cache import conditioning_identity

GOOD=dict(passed=True,clearly_passing=True,resolved_violation=False)
BAD=dict(passed=False,clearly_passing=False,resolved_violation=True)
UNCERTAIN=dict(passed=False,clearly_passing=False,resolved_violation=False)
COMPONENTS={k:dict(GOOD) for k in ('position','quaternion_coordinates','gripper')}
CONFIG={'quality_gate':{'mode':ACTION_ONLY},'quality':{'task_margin':.05,'suite_margin':.02,'absolute_floor':1e-8,'alpha':.05}}


def test_video_failure_is_logged_but_cannot_block_action_only_acceptance():
    selected=monitor_acceptance(CONFIG,GOOD,BAD,COMPONENTS)
    assert selected['passed'] and not selected['resolved_violation']
    assert selected['action_video_diagnostic']['resolved_violation']
    assert not selected['video_is_acceptance_gate']
    assert not monitor_acceptance({},GOOD,BAD,COMPONENTS)['passed']
    # The actual full-check combination and capacity decision see the action policy.
    result=combined_quality(selected,GOOD)
    budget=Budget(original=16_394_878_784,attempt=280)
    assert budget.decide(result['passed'],resolved_violation=result['resolved_violation'])=='accept'
    assert not budget.rescue_used


@pytest.mark.parametrize('component',['position','quaternion_coordinates','gripper'])
def test_each_action_component_still_blocks_acceptance_and_requests_rescue(component):
    components=copy.deepcopy(COMPONENTS);components[component]=dict(BAD)
    result=monitor_acceptance(CONFIG,GOOD,GOOD,components)
    assert not result['passed'] and result['resolved_violation']
    budget=Budget(original=16_394_878_784,attempt=280)
    assert budget.decide(result['passed'],resolved_violation=result['resolved_violation'])=='rescue_confirmed_harm_once'


def test_action_uncertainty_stays_uncertain_and_native_action_harm_blocks():
    result=monitor_acceptance(CONFIG,UNCERTAIN,BAD,COMPONENTS)
    assert not result['passed'] and not result['resolved_violation']
    action_good=monitor_acceptance(CONFIG,GOOD,BAD,COMPONENTS)
    result=combined_quality(action_good,BAD)
    assert not result['passed'] and result['resolved_violation']


def test_fitting_action_harm_still_blocks_further_pruning_with_good_monitor():
    budget=Budget(original=16_394_878_784,attempt=140,progress=140)
    decision=budget.observe_fitting(dict(source='fitting_only',complete=True,**BAD),dict(ffn_max=0.,cross_heads_max=0.,hidden=0.))
    assert decision=='hold_fitting_harm' and budget.authorized_updates==0


def test_missing_action_component_or_unknown_policy_is_rejected():
    with pytest.raises(ValueError):monitor_acceptance(CONFIG,GOOD,GOOD,{'position':GOOD})
    for entry in [{'mode':'video_only'}, {'mode':ACTION_ONLY,'override_pass':True}, ACTION_ONLY, None]:
        with pytest.raises(ValueError):policy_mode({'quality_gate':entry})
    assert policy_mode({})==LEGACY


def test_action_margins_and_confidence_correction_are_unchanged():
    old=copy.deepcopy(CONFIG);old.pop('quality_gate')
    assert quality_settings(CONFIG)==quality_settings(old)
    assert quality_settings(CONFIG)['task_margin']==.05
    assert quality_settings(CONFIG)['suite_margin']==.02


def test_encoder_cache_reused_but_protocol_remains_in_scientific_configuration():
    old={'seed':100008,'teacher_bundle':'same','quality':CONFIG['quality']}
    new=copy.deepcopy(old);new['quality_gate']={'mode':ACTION_ONLY}
    assert conditioning_identity(old)==conditioning_identity(new)
    assert old!=new
    changed=copy.deepcopy(new);changed['teacher_bundle']='different'
    assert conditioning_identity(changed)!=conditioning_identity(new)


def test_stability_requires_five_checks_under_the_same_acceptance_policy():
    selection={'hidden_indices':[0,1], 'ffn_indices':[[0,1]],'cross_head_indices':[[0]]}
    records=[dict(lineage_update=i*70,cap=100,selection=copy.deepcopy(selection),quality_passed=True,screen_clear=True,gradients_active=True) for i in range(5)]
    assert stable(records)
    records[-1]['quality_gate_mode']=ACTION_ONLY
    assert not stable(records)
    for row in records:row['quality_gate_mode']=ACTION_ONLY
    assert stable(records)
