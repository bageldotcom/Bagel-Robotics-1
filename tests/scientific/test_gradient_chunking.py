import pytest
import torch
from bagel_robotics._internal.training.calibration_batch import measure_in_chunks


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
