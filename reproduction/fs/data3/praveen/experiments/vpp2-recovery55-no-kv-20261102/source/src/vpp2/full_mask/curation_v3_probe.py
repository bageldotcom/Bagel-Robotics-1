"""Native action-gradient coverage on selection versus development episodes.

Final-audit episodes are never read. This measures local sensitivity coverage,
not simulator success, and performs no optimizer updates.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from .panel import seed_for, immutable, file_hash
from .data import collate, prepared
from .runtime import initialize, install, cache, predict, model_versions, check_frozen
from .gpu_prepare import checkpoint_identity
from vpp2.compact_kv.losses import action_error


def sample_rows(panel, task, role, count, seed):
    episodes = defaultdict(list)
    for r in panel['windows']:
        if r['task'] == task and r['role'] == role:
            episodes[r['episode']].append(r)
    if len(episodes) < count:
        raise ValueError('Insufficient independent development/selection episodes')
    rng = np.random.default_rng(seed_for(seed, 'v3-probe', task, role))
    chosen = rng.choice(sorted(episodes), count, replace=False)
    return [episodes[ep][int(rng.integers(len(episodes[ep])))] for ep in chosen]


def similarity(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    k = max(1, a.numel() // 20)
    ia = set(torch.argsort(a, descending=True, stable=True)[:k].tolist())
    ib = set(torch.argsort(b, descending=True, stable=True)[:k].tolist())
    denominator = float(a.norm() * b.norm())
    return dict(top5_jaccard=len(ia & ib) / len(ia | ib) if denominator > 1e-20 else None,
                cosine=float(a @ b) / denominator if denominator > 1e-20 else None,
                selection_norm=float(a.norm()), development_norm=float(b.norm()))


def probe(config, panel, windows_per_role=8, draws=2, *, initialized=None, rank=0, world=1):
    if not 2 <= windows_per_role <= 8 or draws < 1:
        raise ValueError('V3 has eight development episodes per task; use 2..8 independent episodes')
    if initialized is None:
        model, reader, controller = initialize(config);branch=install(model)
    else:model,reader,controller,branch=initialized
    versions=model_versions(model)
    root=Path(config.get('probe_output_dir',config['curation']['output']));root.mkdir(parents=True,exist_ok=True)
    profiles={};records=[]
    tasks = panel['tasks']; roles = ('selection_initial', 'monitor')
    for task in tasks[rank::world]:
        for role in roles:
            chosen = sample_rows(panel, task, role, windows_per_role, config['seed'])
            sums = [(torch.zeros_like(controller.hidden_scores, device='cpu'),
                     torch.zeros_like(controller.ffn_scores, device='cpu')) for _ in range(2)]
            counts = [0, 0]
            for i, row in enumerate(chosen):
                for draw in range(draws):
                    position = -3_000_000 - int(seed_for(config['seed'], row['window_id'], draw) % 1_000_000_000)
                    parts = prepared(model, collate(reader, [row], [position]), config['seed'], [(i + 2 * draw) % 4])
                    h = torch.ones_like(controller.hidden_scores, requires_grad=True)
                    f = torch.ones_like(controller.ffn_scores, requires_grad=True)
                    prediction = predict(model, parts, cache(branch, parts, (h, f)))
                    loss = action_error(prediction, parts[5], parts[0]['action_loss_is_pad']).mean()
                    gh, gf = torch.autograd.grad(loss, (h, f))
                    if not all(torch.isfinite(v).all() for v in (gh, gf, loss)):
                        raise FloatingPointError('Nonfinite native action sensitivity')
                    half = i % 2
                    sums[half][0].add_(gh.detach().float().abs().cpu())
                    sums[half][1].add_(gf.detach().float().abs().cpu()); counts[half] += 1
                    records.append(dict(task=task, role=role, episode=row['episode'], frame=row['frame'],
                                        draw=draw, noise_stratum=(i + 2 * draw) % 4,
                                        gt_action_error=float(loss.detach())))
                    del prediction, loss, gh, gf, h, f, parts
            for kind, j in [('hidden', 0), ('ffn', 1)]:
                profiles[f'{task}/{role}/{kind}'] = (sums[0][j] + sums[1][j]) / sum(counts)
                for half in (0, 1):
                    profiles[f'{task}/{role}/{kind}/half{half}'] = sums[half][j] / counts[half]
            print(json.dumps(dict(event='coverage_probe_v3', task=task, role=role,
                                  completed_draws=len(records), total_draws=len(tasks)*2*windows_per_role*draws)), flush=True)
    check_frozen(model, versions)
    if world>1:
        from .runtime import barrier
        torch.save(dict(profiles=profiles,records=records),root/f'probe_rank_{rank:02d}.pt')
        barrier()
        if rank!=0:
            barrier();return
        profiles={};records=[]
        for worker in range(world):
            shard=torch.load(root/f'probe_rank_{worker:02d}.pt',map_location='cpu',weights_only=True)
            if set(profiles)&set(shard['profiles']):raise ValueError('Duplicated task profiles across ranks')
            profiles.update(shard['profiles']);records.extend(shard['records'])
    comparisons = {}; suite_tasks = defaultdict(list)
    for task in tasks:
        suite_tasks[panel['coverage'][task]['suite']].append(task)
        a = profiles[f'{task}/selection_initial/hidden']; b = profiles[f'{task}/monitor/hidden']
        af = profiles[f'{task}/selection_initial/ffn']; bf = profiles[f'{task}/monitor/ffn']
        comparisons[task] = dict(hidden=similarity(a, b), ffn_by_layer=[similarity(x, y) for x, y in zip(af, bf)],
            within_selection_hidden=similarity(profiles[f'{task}/selection_initial/hidden/half0'], profiles[f'{task}/selection_initial/hidden/half1']),
            within_development_hidden=similarity(profiles[f'{task}/monitor/hidden/half0'], profiles[f'{task}/monitor/hidden/half1']))
    suites = {}
    for suite, names in suite_tasks.items():
        # Equal task weighting, independent of task episode length or sample count.
        vals = {role: [torch.stack([profiles[f'{t}/{role}/{kind}'] for t in names]).mean(0)
                       for kind in ('hidden', 'ffn')] for role in roles}
        suites[suite] = dict(tasks=names, hidden=similarity(vals[roles[0]][0], vals[roles[1]][0]),
            ffn_by_layer=[similarity(x, y) for x, y in zip(vals[roles[0]][1], vals[roles[1]][1])])
    path = root / 'model_probe_profiles.pt'; temp = path.with_suffix('.partial')
    torch.save(profiles, temp); temp.replace(path)
    receipt = dict(passed=True, panel_identity=panel['identity'], checkpoints=checkpoint_identity(config),
                   profiles_sha256=file_hash(path), windows_per_role=windows_per_role, draws=draws,
                   rows=records, comparisons=comparisons, suite_comparisons=suites,
                   frozen_native_weights=True, optimizer_steps=0, audit_episodes_used=0,
                   policy_retention_established=False, coverage_sufficiency_established=False,
                   interpretation='Execution and finite-gradient checks passed. Compare cross-split agreement with within-split variability before interpreting coverage. No policy-retention claim.')
    immutable(root / 'model_probe_receipt.json', receipt)
    if world>1:barrier()
