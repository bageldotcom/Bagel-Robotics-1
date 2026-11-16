import copy
import numpy as np
import pytest
from vpp2.critical_windows.contracts import (SCHEMA, critical_mask, frame_interval,
                                          cue_visible, validate_annotation)
from vpp2.critical_windows.metrics import (native_queries, executed_mask, action_errors,
                                         score_chunk, aggregate)
from vpp2.critical_windows.coverage import coverage_for_episode


def interval(a=10, b=13, kind='precision_alignment'):
    return dict(id='alignment-1', frames=[a,b], kind=kind, label='Align before insertion',
                evidence_frames=[a], evidence='Tip aligned with visible opening.')


def test_native_queries_cover_every_executed_step_once_including_tail():
    for length in (1, 24, 25, 32, 47, 57, 100):
        counts = np.zeros(length, dtype=int)
        for start in native_queries(length):
            offsets = np.nonzero(executed_mask(start, length))[0]
            counts[start + offsets] += 1
        assert (counts == 1).all()


def test_critical_steps_not_whole_chunk_and_no_double_count():
    mask = critical_mask(0, 32, 50, [interval(), interval(11,14)])
    assert np.nonzero(mask)[0].tolist() == [10,11,12,13]
    # A future critical step that will be discarded by replanning is not scored.
    scores = score_chunk(0, 50, {'mse': np.ones(32)},
                         {'status':'verified','intervals':[interval(28,31)]})
    assert scores['groups']['critical_executed']['count'] == 0


def test_unknown_annotations_are_not_a_zero_critical_error():
    result = score_chunk(0, 40, {'mse':np.ones(32)})
    assert result['critical_labels_available'] is False
    assert 'critical_executed' not in result['groups']
    with pytest.raises(ValueError, match='Draft'):
        score_chunk(0, 40, {'mse':np.ones(32)}, {'status':'draft','intervals':[interval()]})


def test_final_steps_and_interval_exclusive_end():
    assert critical_mask(48,32,57,[interval(54,57)]).sum() == 3
    assert critical_mask(0,32,57,[interval(32,35)]).sum() == 0
    for pair in ([0.0,3], [3,3], [-1,4], [0,51], [False,3]):
        with pytest.raises(ValueError): frame_interval(pair,50)


def test_quaternion_sign_invariance_and_invalid_prediction_is_penalized():
    gt = np.zeros((32,16));gt[:,3]=1;gt[:,11]=1
    pred = gt.copy();pred[:,3]=-1;pred[:,11]=-1;pred[:,0]=.001
    errors = action_errors(pred,gt,pred,gt)
    assert np.allclose(errors['left_translation_mm'],1)
    assert np.allclose(errors['left_rotation_degrees'],0)
    pred[:,3:7]=0
    errors = action_errors(pred,gt,pred,gt)
    assert (errors['left_rotation_degrees']==180).all()
    assert (errors['left_invalid_rotation']==1).all()
    pred[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):action_errors(pred,gt,pred,gt)


def test_memory_cue_must_be_in_actual_native_history():
    x=interval(100,110,'memory_decision');x['cue_status']='identified';x['cue_intervals']=[[24,26]]
    assert cue_visible(100,x) is True
    x['cue_intervals']=[[30,35]]
    assert cue_visible(100,x) is False
    x['cue_intervals']=[[0,1],[30,35]]
    assert cue_visible(100,x) is False
    x['cue_status']='uncertain'
    assert cue_visible(100,x) is None


def test_phase_coverage_uses_union_and_keeps_missing_intervals_visible():
    ep=dict(length=100,episode=1,task='task',suite='precision',role='fit')
    rows=[dict(action_interval=[0,32],frame=0,window_id='w1'),
          dict(action_interval=[32,64],frame=32,window_id='w2')]
    a=dict(intervals=[interval(28,40),dict(interval(80,90),id='other')])
    result=coverage_for_episode(ep,rows,a)
    assert result['critical_steps']==22
    assert result['covered_critical_steps']==12
    assert result['intervals'][0]['fully_covered']
    assert result['intervals'][1]['coverage_fraction']==0


def annotation_fixture():
    ep=dict(episode=1,task='task',role='fit',length=50)
    manifest=dict(identity='manifest',episodes=[ep])
    packet=dict(identity='packet',episode=1,manifest_identity='manifest',coarse_pages=[{'file':'coarse.jpg'}])
    value=dict(schema=SCHEMA,manifest_identity='manifest',packet_identity='packet',**ep,
               status='verified',reviewer='local visual reviewer',full_timeline_inspected=True,
               inspected_coarse_pages=['coarse.jpg'],inspected_dense_pages=[{'file':'dense.jpg','sha256':'example'}],intervals=[interval()])
    return manifest,packet,value


def test_annotation_requires_evidence_and_cannot_open_other_episode():
    m,p,a=annotation_fixture()
    assert validate_annotation(m,a,p)['usable']
    for field,new in [('reviewer',''),('full_timeline_inspected',False),
                      ('inspected_coarse_pages',[]),('episode',999),('role','audit')]:
        bad=copy.deepcopy(a);bad[field]=new
        with pytest.raises(ValueError):validate_annotation(m,bad,p)
    a['status']='unreviewed'
    assert not validate_annotation(m,a,p)['usable']


def test_aggregation_is_episode_balanced_and_rejects_duplicate_queries():
    def record(ep,frame,value):
        return dict(variant='native',episode=ep,frame=frame,task='task',suite='precision',
                    scores=score_chunk(frame,72,{'mse':np.full(32,value)}))
    rows=[record(1,0,1),record(1,24,1),record(2,0,3)]
    r=aggregate(rows)
    assert r['by_task'][0]['metrics']['mse']==2
    assert r['by_suite'][0]['metrics']['mse']==2
    with pytest.raises(ValueError,match='Duplicate'):aggregate(rows+[rows[0]])
