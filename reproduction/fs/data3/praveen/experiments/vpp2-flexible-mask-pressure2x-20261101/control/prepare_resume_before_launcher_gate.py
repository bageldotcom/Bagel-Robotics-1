"""Promote a measured execution candidate, otherwise resume unchanged production."""
import copy
import json
import os
import shutil
import sys
from pathlib import Path

from vpp2.flexible_mask.config import load, identity, sources
from vpp2.flexible_mask.continuation import scientific_recipe, validate_parent
from vpp2.flexible_mask.preflight import cpu, require_receipt
from vpp2.full_mask.panel import file_hash
from vpp2.compact_kv.spec import atomic_json

REVIEW = Path('/data3/praveen/experiments/vpp2-flexible-mask-execution-review-20261101')
PARENT = Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261031-speed4-micro4')
TARGET = Path('/data3/praveen/experiments/vpp2-flexible-mask-recovery-20261101-speed5')


def verify_manifest(root):
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if file_hash(Path(manifest['source']) / name) != expected:
            raise ValueError('Published source changed: ' + name)
    return manifest


def promote():
    review = load(REVIEW / 'config.json')
    decision_path = REVIEW / 'checks/execution_decision.json'
    if not decision_path.exists():
        return dict(root=str(PARENT), selected='baseline', reason='No complete benchmark decision')
    decision = json.loads(decision_path.read_text())
    if decision['identity'] != identity(review):
        raise ValueError('Benchmark identity mismatch')
    if decision['selected'] == 'baseline':
        return dict(root=str(PARENT), selected='baseline', reason='No candidate passed speed and numerical checks',
                    decision_sha256=file_hash(decision_path))
    if decision['selected'] not in ('communication', 'selective', 'combined'):
        raise ValueError('Unexpected candidate')
    if decision['source_files_sha256'] != file_hash(REVIEW / 'source_manifest.json'):
        raise ValueError('Source manifest changed')
    manifest = verify_manifest(REVIEW)
    if manifest['files'] != sources():
        raise ValueError('Promotion is using different source')
    for name, expected in decision['evidence'].items():
        if file_hash(REVIEW / name) != expected:
            raise ValueError('Benchmark evidence changed: ' + name)
    result = decision['results'][decision['selected']]
    if not (result['passed'] and result['numerics_passed'] and result['time_reduction'] >= .15):
        raise ValueError('Candidate did not meet fixed adoption criteria')
    require_receipt(review, 'cpu')
    _, receipt = validate_parent(review, 'mask')
    if receipt['update'] != 556:
        raise ValueError('Unexpected checkpoint')
    reports = [json.loads((REVIEW / 'execution_review' / decision['selected'] / f'rank{i}.json').read_text())
               for i in range(8)]
    if not all(r['start'] == 556 and r['end'] == 560 and r['contracts']['passed'] and
               r['frozen_unchanged'] and r['global_batch'] == 256 and r['micro_batch'] == 4 and
               r['execution'] == result['execution'] and r['identity'] == identity(review) for r in reports):
        raise ValueError('Incomplete distributed execution evidence')
    config = copy.deepcopy(review)
    config['root'] = str(TARGET)
    config['execution'] = result['execution']
    config['wandb']['name'] = 'GT1-video005-lateKV005-flexible-cap1-' + decision['selected'] + '-verified'
    if scientific_recipe(config) != scientific_recipe(load(PARENT / 'config.json')):
        raise ValueError('Scientific recipe changed')
    for directory in ('checks', 'control', 'logs', 'mask/diagnostics'):
        (TARGET / directory).mkdir(parents=True, exist_ok=True)
    if (TARGET / 'config.json').exists() and json.loads((TARGET / 'config.json').read_text()) != config:
        raise ValueError('Target already contains a different experiment')
    atomic_json(TARGET / 'config.json', config)
    atomic_json(TARGET / 'source_manifest.json', manifest)
    inputs = json.loads((REVIEW / 'inputs.json').read_text())
    inputs['identity'] = identity(config)
    atomic_json(TARGET / 'inputs.json', inputs)
    validate_parent(config, 'mask')
    cpu(config)
    proof = []
    # Early checkpoints have no ready-for-recovery candidates; keep full history
    # in client state and copy every metric/diagnostic, without duplicating shards.
    paths = [PARENT / 'mask/metrics.jsonl']
    paths += sorted((PARENT / 'mask/diagnostics').rglob('*'))
    paths += sorted((PARENT / 'mask').glob('masks_*.pt'))
    for source in paths:
        if not source.is_file():
            continue
        destination = TARGET / 'mask' / source.relative_to(PARENT / 'mask')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_hash(destination) != file_hash(source):
                raise ValueError('Continuation history changed: ' + str(destination))
        else:
            shutil.copyfile(source, destination)
        proof.append(dict(source=str(source), destination=str(destination), sha256=file_hash(destination)))
    atomic_json(TARGET / 'control/continuation_history.json', dict(parent=str(PARENT), update=556, files=proof))
    verification = dict(
        identity=identity(config), passed=True, world_size=8, micro_batch=4, global_batch=256,
        benchmark_executed_identity=identity(review), benchmark_execution=result['execution'],
        benchmark_variant=decision['selected'], benchmark_start=556, benchmark_end=560,
        benchmark_updates_discarded=True, continuation_checkpoint=config['resume_from'],
        numerical_comparison_receipt=str(decision_path), numerical_comparison_sha256=file_hash(decision_path),
        frozen_reference_unchanged=True, contracts=True, export_equivalence_still_required=True,
        execution_restart_after_new_checkpoint_not_separately_benchmarked=True,
        parent_restart_and_optimizer_state_compared_across_variants=True,
        source_manifest_sha256=file_hash(REVIEW / 'source_manifest.json'),
        peak_allocated_gib=result['peak_allocated_gib'])
    atomic_json(TARGET / 'checks/gpu.json', verification)
    atomic_json(TARGET / 'control/promotion.json', dict(identity=identity(config), passed=True,
        resume_update=556, benchmark_updates_discarded=True, verification=verification))
    return dict(root=str(TARGET), selected=decision['selected'], identity=identity(config),
                decision_sha256=file_hash(decision_path), time_reduction=result['time_reduction'])


def main():
    chosen_path = REVIEW / 'control/production_selection.json'
    if chosen_path.exists():
        chosen = json.loads(chosen_path.read_text())
    else:
        try:
            chosen = promote()
        except Exception as error:
            # A failed optimization check never blocks the previously verified recipe.
            chosen = dict(root=str(PARENT), selected='baseline', reason='Promotion rejected', error=repr(error))
            print('Optimization promotion rejected: ' + repr(error), file=sys.stderr, flush=True)
        chosen['checkpoint_frequency'] = 250
        chosen['global_batch'] = 256
        chosen['stopping_criteria_unchanged'] = True
        atomic_json(chosen_path, chosen)
    root = Path(chosen['root'])
    manifest = verify_manifest(root)
    status_path = root / 'mask/status.json'
    finished = False
    if status_path.exists():
        status = json.loads(status_path.read_text())
        finished = ((status['status'] == 'ready_for_recovery' and status['update'] >= 2000) or
                    (status['status'] == 'selection_not_converged' and status['update'] == 3000))
        if not finished and status['status'] != 'checkpointed_interruption':
            raise ValueError('Cannot restart failed or unresolved training')
    output = dict(chosen, config=str(root / 'config.json'), source=manifest['source'], finished=finished)
    atomic_json(REVIEW / 'control/launch_selection.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
