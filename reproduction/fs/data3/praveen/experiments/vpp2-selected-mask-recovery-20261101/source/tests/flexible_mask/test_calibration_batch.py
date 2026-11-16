import pytest
import torch
from vpp2.flexible_mask.calibration_batch import measure_in_chunks


def toy_gradients(rows, positions):
    x=torch.tensor(rows,dtype=torch.float32)
    masks=[torch.tensor([.2,-.3,.7],requires_grad=True) for _ in range(3)]
    values={key:sum(((x*p).sin()+p.square()).square().mean()*(i+1) for i,p in enumerate(masks)) for key in ('gt','video','kv')}
    return {key:[g.detach() for g in torch.autograd.grad(loss,masks)] for key,loss in values.items()}


@pytest.mark.parametrize('count,chunk',[(4,2),(5,2),(2,2),(4,1),(1,2)])
def test_matches_complete_mean_gradient(count,chunk):
    rows=[[i*.13-.5,.2-i*.09,.8] for i in range(count)]
    expected=toy_gradients(rows,list(range(count)))
    actual=measure_in_chunks(rows,list(range(count)),chunk,toy_gradients)
    for key in expected:
        for x,y in zip(expected[key],actual[key]):torch.testing.assert_close(x,y,rtol=2e-6,atol=1e-7)


def test_averages_signed_vectors_before_norm():
    def call(rows,pos):return {key:[torch.tensor(rows).float().mean().reshape(1) for _ in range(3)] for key in ('gt','video','kv')}
    result=measure_in_chunks([1.,-1.],[0,1],1,call)
    assert all(torch.equal(g,torch.zeros_like(g)) for values in result.values() for g in values)


def test_preserves_sample_and_position_pairing():
    calls=[]
    def call(rows,pos):
        calls.append((list(rows),list(pos)));return toy_gradients(rows,pos)
    rows=[[1.,2.,3.]]*5
    measure_in_chunks(rows,[10,11,12,13,14],2,call)
    assert [p for _,part in calls for p in part]==[10,11,12,13,14]
    assert [len(r) for r,_ in calls]==[2,2,1]


@pytest.mark.parametrize('rows,positions,chunk',[([],[],2),([1],[1,2],2),([1],[1],0)])
def test_invalid_inputs(rows,positions,chunk):
    with pytest.raises(ValueError):measure_in_chunks(rows,positions,chunk,lambda *args:{})


def test_rejects_retained_graph():
    def call(rows,pos):return {k:[torch.ones(1,requires_grad=True) for _ in range(3)] for k in ('gt','video','kv')}
    with pytest.raises(ValueError,match='detached'):measure_in_chunks([1],[0],2,call)


def test_rejects_changed_inventory():
    with pytest.raises(ValueError,match='inventory'):measure_in_chunks([1],[0],2,lambda *a:{'gt':[]})


@pytest.mark.parametrize('diagnostic,expected_batches',[(False,[4]),(True,[2,2,4])])
def test_runtime_finishes_small_diagnostics_before_main_graph(monkeypatch,diagnostic,expected_batches):
    from types import SimpleNamespace
    import vpp2.flexible_mask.runtime as module
    runtime=object.__new__(module.Runtime)
    runtime.config={'seed':100008,'execution':{'calibration_micro_batch':2}}
    runtime.engine=SimpleNamespace(_scale_wrt_gas=True);runtime.reader=None;runtime.device=torch.device('cpu');runtime.masks=object()
    runtime.reference=SimpleNamespace(train_action_scheduler=SimpleNamespace(training_weight=lambda t:torch.ones(len(t))))
    runtime.gates=lambda *args:None
    batches=[];measured=[]
    def prepare(ref,reader,rows,positions,seed,strata):
        batches.append(len(rows));b=len(rows)
        return ({'count':b,'action_loss_is_pad':torch.zeros(b,1,dtype=torch.bool)},None,None,None,torch.ones(b),torch.zeros(b,1,1),None)
    runtime.video=lambda parts,**kwargs:{'cache':parts[0]['count'],'video':torch.zeros(parts[0]['count'],1,5,1,1)}
    runtime.predict=lambda parts,cache:torch.zeros(parts[0]['count'],1,1)
    def gradients(parts,update,rows,*args):
        measured.extend(rows)
        return {k:[torch.tensor([sum(rows)/len(rows)],dtype=torch.float32) for _ in range(3)] for k in ('gt','video','kv')}
    runtime.measure_gt_mask_gradients=gradients
    monkeypatch.setattr(module,'prepare',prepare)
    monkeypatch.setattr(module,'cache_terms',lambda a,b:torch.zeros(a,40,2))
    monkeypatch.setattr(module,'future_mask',lambda rows,device:torch.ones(len(rows),2))
    monkeypatch.setattr(module,'objectives',lambda *args:'native-main-objective')
    assert runtime.forward([0,1,2,3],[10,11,12,13],193,measure_gt=diagnostic)=='native-main-objective'
    assert batches==expected_batches
    assert runtime.engine._scale_wrt_gas is False
    if diagnostic:
        assert measured==[0,1,2,3]
        assert runtime.measured_mask_gradients['gt'][0].item()==1.5
    else:assert runtime.measured_mask_gradients is None
