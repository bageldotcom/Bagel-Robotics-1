import copy
import pytest
import torch
from vpp2.flexible_mask.runtime import FrozenReplica,independent_storage


def test_resident_teacher_preserves_values_rng_and_freezing():
    module=torch.nn.Linear(8,8).requires_grad_(False).eval()
    x=torch.randn(3,8);rng=torch.get_rng_state().clone()
    teacher=FrozenReplica(module)
    assert torch.equal(teacher(x),module(x))
    assert torch.equal(torch.get_rng_state(),rng)
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    student=copy.deepcopy(module).requires_grad_(True)
    independent_storage(student,teacher.module)
    with pytest.raises(RuntimeError,match='alias'):independent_storage(module,teacher.module)
    with pytest.raises(ValueError,match='frozen'):FrozenReplica(student)


def test_only_teacher_storage_can_change_in_execution_recipe():
    from pathlib import Path
    from vpp2.flexible_mask.config import load,validate
    c=load(Path(__file__).resolve().parents[2]/'configs/flexible_mask/full30k.json')
    for mode in ('sharded','replicated'):
        c['execution']={'teacher_storage':mode};validate(c)
    c['execution']['micro_batch']=2
    with pytest.raises(ValueError,match='execution'):validate(c)
