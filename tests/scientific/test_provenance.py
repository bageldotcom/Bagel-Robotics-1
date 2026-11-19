"""Current-code identities bind scientific settings and permit movable outputs."""
import copy
import json
from pathlib import Path
from bagel_robotics.adaptive_pruning.config import identity,validate


def recipe():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/recipes/mask_learning.json').read_text())


def test_final_recipe_configuration_validates():
    for name in ('mask_learning','recovery'):
        value=json.loads((Path(__file__).resolve().parents[2]/'configs/recipes'/f'{name}.json').read_text())
        validate(value)


def test_output_and_resume_selectors_do_not_change_scientific_identity():
    config=recipe();changed=copy.deepcopy(config)
    changed.update(root='/new/output',resume='/explicit/resume',runtime_conditioning_cache='/new/cache')
    assert identity(config)==identity(changed)


def test_numerical_recipe_change_changes_identity():
    config=recipe();changed=copy.deepcopy(config);changed['train']['video_lr']*=2
    assert identity(config)!=identity(changed)
