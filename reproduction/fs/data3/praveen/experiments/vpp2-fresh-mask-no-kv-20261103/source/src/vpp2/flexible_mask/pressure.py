"""Explicit, bounded size-pressure change; other objectives stay fixed."""
import copy
import json
import math
from pathlib import Path


def validate(config):
    p = config.get('pressure_continuation')
    if p is None:
        return
    keys = {'schema', 'start_update', 'ramp_updates', 'target_multiplier',
            'base_coefficient', 'absolute_cap', 'target_video_parameters',
            'automatic_second_increase'}
    if set(p) != keys or p['schema'] != 'vpp2-size-pressure-ramp-v1':
        raise ValueError('Invalid pressure continuation schema')
    start = p['start_update']
    if type(start) is not int or not 1000 <= start <= 1750:
        raise ValueError('Pressure start needs a checkpoint with time for stability checks')
    if type(p['ramp_updates']) is not int or p['ramp_updates'] != 250 or p['target_multiplier'] != 2.:
        raise ValueError('Only the reviewed 250-update, twofold ramp is enabled')
    if p['automatic_second_increase'] is not False:
        raise ValueError('Automatic additional pressure increases are disabled')
    if p['target_video_parameters'] != 10_000_000_000:
        raise ValueError('Unexpected reporting target; it is not a hard pruning threshold')
    base, cap = p['base_coefficient'], p['absolute_cap']
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (base, cap)):
        raise ValueError('Pressure coefficients must be finite and positive')
    if not math.isclose(cap, 2 * base, rel_tol=1e-12, abs_tol=0):
        raise ValueError('Pressure cap must equal the twofold coefficient')
    if cap > config['size_coefficient']['upper_cap']:
        raise ValueError('Pressure exceeds the existing safety cap')


def weight(coefficient, config, update):
    if config.get("regularization_replay"):
        from .replay import weight as replay_weight
        return replay_weight(coefficient,config,update)
    baseline = coefficient.weight(update)
    p = config.get('pressure_continuation')
    if p is None:
        return baseline
    validate(config)
    if (coefficient.applied is None or coefficient.unresolved or
        not math.isclose(coefficient.applied, p['base_coefficient'], rel_tol=1e-12, abs_tol=0)):
        raise ValueError('Pressure does not match the saved calibrated coefficient')
    if coefficient.resolved_update is None or coefficient.resolved_update + 100 > p['start_update']:
        raise ValueError('Original calibration must finish before the pressure ramp')
    progress = min(max((update - p['start_update']) / p['ramp_updates'], 0.), 1.)
    return min(baseline * (1 + progress * (p['target_multiplier'] - 1)), p['absolute_cap'])


def comparable_recipe(config):
    result = copy.deepcopy(config)
    for key in ('root', 'resume_from', 'pressure_continuation'):
        result.pop(key, None)
    result['wandb'].pop('name', None)
    return result


def validate_recipe_change(config, parent):
    validate(config)
    if 'pressure_continuation' in parent:
        raise ValueError('Stacking pressure continuations is not authorized')
    if comparable_recipe(config) != comparable_recipe(parent):
        raise ValueError('Pressure continuation changed another scientific or execution setting')


def validate_parent_state(state, config):
    p = config['pressure_continuation']
    if state['update'] != p['start_update'] or state['cursor'] != p['start_update'] * 256:
        raise ValueError('Pressure ramp must start at the pinned checkpoint cursor')
    from .coefficient import Coefficient
    coefficient = Coefficient.from_state(state['coefficient'], config['size_coefficient']['upper_cap'])
    weight(coefficient, config, state['update'])


def require_runtime_evidence(config):
    """Reuse GPU evidence only for unchanged runtime, not as a newly executed trial."""
    from .config import identity, sources
    from .preflight import require_receipt
    from vpp2.full_mask.panel import file_hash
    root = Path(config['root'])
    require_receipt(config, 'cpu')
    proof = json.loads((root / 'checks/pressure_prerequisites.json').read_text())
    if proof['identity'] != identity(config) or proof['passed'] is not True:
        raise ValueError('Stale pressure prerequisites')
    for item in proof['evidence']:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Changed pressure evidence: ' + item['path'])
    parent = json.loads(Path(proof['parent_manifest']).read_text())['files']
    current = published_sources(config)
    allowed = {
        'src/vpp2/flexible_mask/pressure.py', 'src/vpp2/flexible_mask/config.py',
        'src/vpp2/flexible_mask/continuation.py', 'src/vpp2/flexible_mask/train.py',
        'tests/flexible_mask/test_pressure_continuation.py',
    }
    changed = {k for k in parent.keys() | current.keys() if parent.get(k) != current.get(k)}
    if not changed or not changed <= allowed or sorted(changed) != proof['changed_source_files']:
        raise ValueError('Pressure changes exceed the reviewed source scope')
    previous = json.loads(Path(proof['parent_gpu_receipt']).read_text())
    if previous['passed'] is not True or previous['identity'] != config['resume_from']['identity']:
        raise ValueError('Original GPU runtime verification is missing')
    return proof


def verify_gpu(runtime, config, state, coefficient):
    """Small mask-gradient check; no policy updates, random draws or optimizer changes."""
    import torch
    import torch.distributed as dist
    from .config import identity
    from .preflight import fingerprint
    from .rng import rng_state
    from vpp2.compact_kv.spec import atomic_json
    require_runtime_evidence(config)
    if state is None or state['update'] < config['pressure_continuation']['start_update']:
        raise ValueError('Pressure verification requires a resumed checkpoint')
    before = fingerprint((runtime.masks.state_dict(), runtime.mask_optimizer.state_dict(), rng_state()))
    families = runtime.masks.families()
    old_grad = [None if x.grad is None else x.grad.detach().clone() for x in families]
    start = config['pressure_continuation']['start_update']
    cost = runtime.cost.loss(runtime.masks.gates(state['update']))
    initial, final = weight(coefficient, config, start), weight(coefficient, config, start + 250)
    a = torch.autograd.grad(initial * cost, families, retain_graph=True)
    b = torch.autograd.grad(final * cost, families)
    errors = []
    for left, right in zip(a, b, strict=True):
        if not torch.isfinite(left).all() or not torch.isfinite(right).all() or left.norm() <= 0:
            raise RuntimeError('Nonfinite or inactive pressure gradient')
        error = float((right - 2 * left).norm() / (2 * left.norm()).clamp_min(1e-30))
        if error > 2e-5:
            raise RuntimeError('Pressure gradient did not scale by two')
        errors.append(error)
    for parameter, previous in zip(families, old_grad, strict=True):
        if previous is None and parameter.grad is not None:
            raise RuntimeError('Pressure verification mutated accumulation buffers')
        if previous is not None and not torch.equal(previous, parameter.grad):
            raise RuntimeError('Pressure verification mutated accumulated gradients')
    after = fingerprint((runtime.masks.state_dict(), runtime.mask_optimizer.state_dict(), rng_state()))
    if after != before:
        raise RuntimeError('Pressure verification mutated masks, optimizer or RNG')
    reports = [None] * runtime.world
    dist.all_gather_object(reports, dict(rank=runtime.rank, errors=errors,
                                       mask_state=fingerprint(runtime.masks.state_dict())))
    if len({r['mask_state'] for r in reports}) != 1:
        raise RuntimeError('Replicated masks disagree across ranks')
    if runtime.rank == 0:
        atomic_json(Path(config['root']) / 'checks/pressure_gpu.json', dict(
            identity=identity(config), passed=True, update=state['update'], reports=reports,
            resumed_optimizer_and_rng=True, no_policy_updates_in_probe=True,
            runtime_gpu_evidence_inherited=True, objective_change='size pressure only',
            baseline_weight=initial, final_weight=final))
    dist.barrier()


def published_sources(config):
    """Include non-code artifacts tracked by the original publication manifest."""
    from .config import sources
    from vpp2.full_mask.panel import file_hash
    manifest = json.loads((Path(config['root']) / 'source_manifest.json').read_text())
    source = Path(manifest['source'])
    if source.resolve() != Path(__file__).resolve().parents[3]:
        raise ValueError('Published source directory differs from the running package')
    files = manifest['files']
    current = sources()
    if any(files.get(name) != sha for name, sha in current.items()):
        raise ValueError('Published code manifest is incomplete or changed')
    for name, sha in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_hash(source / relative) != sha:
            raise ValueError('Published source artifact changed: ' + name)
    return files
