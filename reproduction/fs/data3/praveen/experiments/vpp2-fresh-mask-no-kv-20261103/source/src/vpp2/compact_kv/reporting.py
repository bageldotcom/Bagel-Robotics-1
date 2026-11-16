"""Durable loss records and actual-data plots; all layer indices are zero based."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot_log(path, output=None, kv_only=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A reader can encounter a still-being-written final line.
        if row.get('kind') == 'train':
            records.append(row)
    if not records:
        return []
    # Last successful write wins on an exactly resumed optimizer update.
    records = list({r['update']: r for r in records}.values())
    records.sort(key=lambda r: r['update'])
    output = Path(output or path.parent / 'plots')
    output.mkdir(parents=True, exist_ok=True)
    steps = np.array([r['update'] for r in records])
    saved = []

    def save(fig, name):
        destination = output / name
        fig.savefig(destination.with_suffix('.partial.png'), dpi=150, bbox_inches='tight')
        destination.with_suffix('.partial.png').replace(destination)
        plt.close(fig)
        saved.append(str(destination))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fields = [('action_gt', 'Ground-truth action flow loss'),
              ('action_teacher', 'Teacher action-matching loss'),
              ('kv_raw', 'K/V loss: mean of all 40 layers'),
              ('kv_weighted', 'Weighted K/V contribution'),
              ('loss', 'Total training objective'), ('kv_weight', 'Scheduled K/V weight')]
    for ax, (key, label) in zip(axes.flat, fields):
        values = np.array([r[key] for r in records])
        ax.plot(steps, values, alpha=.4, lw=.7, label='Every update')
        width = min(20, len(values))
        if width > 1:
            ax.plot(steps[width-1:], np.convolve(values, np.ones(width)/width, mode='valid'),
                    lw=1.6, label=f'{width}-update mean')
        ax.set(title=label, xlabel='Optimizer update', ylabel='Loss' if key != 'kv_weight' else 'Coefficient')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle(f'{path.parent.name}: individual objectives (native action timestep weighting retained)')
    fig.tight_layout()
    save(fig, 'objectives.png')

    layers = np.array([r['kv_by_layer_kv'] for r in records])
    if layers.shape[1:] != (40, 2):
        raise ValueError('Logs must contain separate K and V losses at every original layer')
    positive = layers[layers > 0]
    vmin = max(float(np.min(positive)) if positive.size else 1e-8, 1e-12)
    vmax = max(float(layers.max()), vmin * 1.01)
    fig, axes = plt.subplots(1, 2, figsize=(17, 10), sharey=True)
    # pcolormesh handles irregular update spacing, including resumed runs.
    edges = np.r_[steps[0] - .5, (steps[1:] + steps[:-1])/2, steps[-1] + .5]
    for j, ax in enumerate(axes):
        mesh = ax.pcolormesh(edges, np.arange(41) - .5, np.maximum(layers[:, :, j].T, vmin),
                             norm=LogNorm(vmin, vmax), cmap='viridis', shading='flat')
        ax.set(title=f'{"K" if j == 0 else "V"}: normalized MSE by original layer', xlabel='Optimizer update', ylabel='Layer index (zero based)')
        ax.set_yticks(range(40), [f'{i} *' if i in kv_only else str(i) for i in range(40)], fontsize=7)
    fig.colorbar(mesh, ax=axes, label='Unweighted teacher-energy-normalized MSE (log color scale)', shrink=.85)
    fig.suptitle('* marks K/V-only blocks. Every layer contributes equally; K and V have equal weight.')
    save(fig, 'kv_layer_heatmaps.png')

    fig, axes = plt.subplots(8, 5, figsize=(21, 24))
    for i, ax in enumerate(axes.flat):
        ax.plot(steps, layers[:, i, 0], lw=.8, label='K')
        ax.plot(steps, layers[:, i, 1], lw=.8, label='V')
        ax.set(title=f'Layer {i}' + (' — K/V-only' if i in kv_only else ' — full block'), xlabel='Update', ylabel='Normalized MSE')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.tight_layout()
    save(fig, 'kv_all_40_layers.png')
    return saved


class Reporter:
    def __init__(self, directory, config, candidate, enabled=True):
        from .spec import CANDIDATES
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'metrics.jsonl'
        self.config, self.candidate, self.kv_only = config, candidate, CANDIDATES[candidate]
        self.wandb = None
        if enabled and config['wandb']['enabled']:
            import wandb
            from .spec import stable_seed
            wc = config['wandb']
            self.wandb = wandb.init(entity=wc['entity'], project=wc['project'], group=wc['group'],
                    name=f'{candidate}-precision-1000', id=f'ckv-{candidate.lower()}-{stable_seed(config["seed"], str(directory)):x}',
                    resume='allow', dir=str(self.directory), config=config)
            wandb.define_metric('update')
            wandb.define_metric('*', step_metric='update')
            (self.directory / 'wandb_url.txt').write_text(self.wandb.url + '\n')

    def write(self, row):
        with self.path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
        if self.wandb:
            values = {'update': row['update']}
            for key, value in row.items():
                if isinstance(value, (int, float)) and key != 'update':
                    values[f'{row["kind"]}/{key}'] = value
            if 'kv_by_layer_kv' in row:
                for i, (k, v) in enumerate(row['kv_by_layer_kv']):
                    values[f'{row["kind"]}/kv_layer/{i:02d}/k'] = k
                    values[f'{row["kind"]}/kv_layer/{i:02d}/v'] = v
                    values[f'{row["kind"]}/kv_layer/{i:02d}/mean'] = (k + v)/2
            for plane, layers in enumerate(row.get('kv_by_plane_layer_kv', [])):
                for i, (k, v) in enumerate(layers):
                    values[f'{row["kind"]}/plane_{plane}/layer_{i:02d}/k'] = k
                    values[f'{row["kind"]}/plane_{plane}/layer_{i:02d}/v'] = v
            for task, metrics in row.get('by_task', {}).items():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        values[f'{row["kind"]}/{task}/{key}'] = value
            self.wandb.log(values)
        if row['kind'] == 'train' and row['update'] % self.config['plot_every'] == 0:
            self.plots()

    def plots(self):
        result = plot_log(self.path, kv_only=self.kv_only)
        if self.wandb and result:
            import wandb
            self.wandb.log({f'plots/{Path(p).stem}': wandb.Image(p) for p in result})
        return result

    def close(self):
        self.plots()
        if self.wandb:
            self.wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log')
    parser.add_argument('--candidate', choices=['E1', 'E2', 'W1', 'W2'], required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    from .spec import CANDIDATES
    print(json.dumps(plot_log(args.log, args.output, CANDIDATES[args.candidate]), indent=2))


if __name__ == '__main__':
    main()
