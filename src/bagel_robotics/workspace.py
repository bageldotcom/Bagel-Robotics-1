"""Portable workspaces, input checks and explicit training qualification."""
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

from .configuration import RESOURCES, read_json, recipe, resolve, resource, selected_recipe


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def contained(root, logical):
    path = PurePosixPath(logical)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute logical path without traversal')
    root = Path(root).resolve()
    target = root.joinpath(*path.parts[1:])
    if not target.resolve().is_relative_to(root):
        raise ValueError('Path escapes workspace: ' + logical)
    return target


def workspace_path(config):
    path = resolve(config, config.get('workspace'))
    if path is None:
        raise ValueError('Set workspace in your research configuration')
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def resolved_recipe(config, stage):
    name = selected_recipe(config, stage)
    value = copy.deepcopy(recipe(name))
    overrides = config.get('recipe_overrides', {}).get(name, {})
    # Explicit merge permits normal research settings without changing the
    # supplied defaults. The scientific validator checks the resulting recipe.
    def merge(target, change):
        for key, item in change.items():
            if isinstance(item, dict) and isinstance(target.get(key), dict):
                merge(target[key], item)
            else:
                target[key] = item
    merge(value, overrides)
    value.update(root='/bagel/workspace/' + name,
                 demonstrations='/bagel/inputs/demonstrations',
                 teacher_bundle='/bagel/inputs/base_model',
                 wan_root='/bagel/inputs/encoders',
                 panel_file='/bagel/workspace/inputs/panel.json',
                 initial_masks='/opt/bagel/src/bagel_robotics/resources/initial_rankings.pt')
    if config.get('paths', {}).get('initial_rankings'):
        value['initial_masks'] = '/bagel/inputs/initial_rankings'
    if name == 'recovery':
        value['compact_model'] = '/bagel/inputs/compact_model'
    if config.get('paths', {}).get('resume'):
        value['resume'] = '/bagel/inputs/resume'
    if config.get('execution', {}).get('engine', 'native') == 'native':
        value['root'] = str(workspace_path(config) / name)
        value['panel_file'] = str(workspace_path(config) / 'inputs/panel.json')
        value['initial_masks'] = str(resolve(config, config.get('paths', {}).get('initial_rankings')) or resource('initial_rankings.pt'))
        for key, field in [('demonstrations', 'demonstrations'), ('base_model', 'teacher_bundle'),
                           ('encoders', 'wan_root'), ('compact_model', 'compact_model'), ('resume', 'resume')]:
            if key == 'compact_model' and name != 'recovery':
                continue
            path = resolve(config, config.get('paths', {}).get(key))
            if path:
                value[field] = str(path)
    return value


def init(config):
    import gzip
    root = workspace_path(config)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a new empty workspace: ' + str(root))
    root.mkdir(parents=True, exist_ok=True)
    (root / 'inputs').mkdir()
    (root / 'inputs/panel.json').write_bytes(gzip.decompress(resource('selected_panel.json.gz').read_bytes()))
    shutil.copy2(resource('normalization.json'), root / 'inputs/normalization.json')
    for stage in ('learn-masks', 'recover'):
        name = selected_recipe(config, stage)
        write_json(root / name / 'config.json', resolved_recipe(config, stage))
    write_json(root / 'workspace.json', {'schema': 'bagel-workspace-v1', 'qualified': False})
    return {'workspace': str(root), 'initialized': True, 'model_weights_included': False}


def bindings(config, stage):
    pins = read_json(resource('artifacts.json'))
    data = read_json(resource('data.json'))
    needs = ['demonstrations', 'encoders', 'base_model']
    if stage == 'recover':
        needs.append('compact_model')
    elif config.get('paths', {}).get('initial_rankings'):
        needs.append('initial_rankings')
    if config.get('paths', {}).get('resume'):
        needs.append('resume')
    catalog = {
        'demonstrations': {'required_files': ['manifest.json', 'dataset_stats.json'],
            'metadata_sha256': {'manifest.json': data['manifest_sha256'], 'dataset_stats.json': data['normalization_sha256']}},
        'base_model': {'required_files': ['video.pt', 'action.pt', 'dataset_stats.json'],
            'metadata_sha256': {'dataset_stats.json': data['normalization_sha256']},
            'weight_sha256': {Path(key).name: value for key, value in pins['training_base']['files'].items()}},
        'encoders': {'required_files': [entry['rfilename'].removeprefix('checkpoints/Wan2.1-I2V-14B-480P/')
                                      for entry in pins['inference']['wan']['files']]},
        'compact_model': {'required_files': []},
        'initial_rankings': {'sha256': data['initial_masks_sha256']},
        'resume': {'required_files': ['COMPLETE.json']},
    }
    result = {}
    for name in needs:
        record = dict(catalog[name])
        record.update(host_path=resolve(config, config.get('paths', {}).get(name)), container_path='/bagel/inputs/' + name)
        result[name] = record
    return result


def check_asset(name, record, full_hash=False):
    path = record.get('host_path')
    if path is None:
        return ['Set paths.' + name + ' in your configuration']
    path = Path(path)
    if not path.exists():
        return [name + ': missing ' + str(path)]
    errors = []
    for relative in record.get('required_files', []):
        if not (path / relative).is_file():
            errors.append(name + ': missing ' + relative)
    fields = ['metadata_sha256', 'weight_sha256'] if full_hash else ['metadata_sha256']
    for field in fields:
        for relative, expected in record.get(field, {}).items():
            item = contained(path, '/' + relative)
            if not item.is_file() or sha256(item) != expected:
                errors.append(name + ': checksum mismatch ' + relative)
    if record.get('sha256') and (not path.is_file() or sha256(path) != record['sha256']):
        errors.append(name + ': checksum mismatch')
    marker = path / 'COMPLETE.json'
    if marker.is_file():
        value = read_json(marker)
        for field in ('identity', 'update'):
            if field in record and value.get(field) != record[field]:
                errors.append(name + ': wrong ' + field)
        for relative, entry in value.get('files', {}).items():
            item = contained(path, '/' + relative)
            if not item.is_file():
                errors.append(name + ': missing ' + relative)
            elif isinstance(entry, dict) and item.stat().st_size != entry.get('size', item.stat().st_size):
                errors.append(name + ': wrong size ' + relative)
            elif full_hash and sha256(item) != (entry['sha256'] if isinstance(entry, dict) else entry):
                errors.append(name + ': checkpoint checksum mismatch ' + relative)
    if name == 'compact_model':
        verified = (path / 'manifest.json').is_file() and (path / 'VERIFIED.json').is_file()
        native = all((path / item).is_file() for item in ('video.pt', 'action.pt', 'compact_architecture.json', 'dataset_stats.json'))
        if not (verified or native):
            errors.append('compact_model: provide a verified compact export or complete native policy bundle')
    return errors


def doctor(config, stage='learn-masks', full_hash=False):
    errors = []
    root = workspace_path(config)
    if not (root / 'workspace.json').is_file():
        errors.append('Initialize the workspace with the init command')
    else:
        pins = read_json(resource('data.json'))
        for relative, expected in [('inputs/panel.json', pins['panel_sha256']),
                                   ('inputs/normalization.json', pins['normalization_sha256'])]:
            path = root / relative
            if not path.is_file() or sha256(path) != expected:
                errors.append('Workspace input checksum differs: ' + relative)
    for key, record in bindings(config, stage).items():
        errors.extend(check_asset(key, record, full_hash))
    current = resolved_recipe(config, stage)
    name = selected_recipe(config, stage)
    staged = root / name / 'config.json'
    if not staged.is_file():
        errors.append('Workspace stage configuration is missing; initialize a new workspace')
    elif read_json(staged) != current:
        errors.append('Workspace recipe differs from configuration; prepare a new workspace for the changed experiment')
    return {'passed': not errors, 'stage': stage, 'errors': errors,
            'scope': 'Configuration and input completeness; no model execution'}


def qualification_fingerprint(config, stage):
    package = Path(__file__).resolve().parent
    source = {str(path.relative_to(package)): sha256(path) for path in package.rglob('*')
              if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.pt', '.gz'} and '__pycache__' not in path.parts}
    assets = {}
    for name, record in bindings(config, stage).items():
        path = record['host_path']
        metadata = {}
        if path and path.exists():
            if path.is_file():
                metadata['file'] = [path.stat().st_size, path.stat().st_mtime_ns]
            else:
                for child in sorted(path.rglob('*')):
                    if child.is_file():
                        metadata[str(child.relative_to(path))] = [child.stat().st_size, child.stat().st_mtime_ns]
        assets[name] = {'path': str(path), 'files': metadata}
    image = resolve(config, config.get('execution', {}).get('image'))
    root = workspace_path(config)
    staged = {relative: sha256(root / relative) if (root / relative).is_file() else None
              for relative in ['inputs/panel.json', 'inputs/normalization.json',
                               selected_recipe(config, stage) + '/config.json']}
    runtime = None
    execution = config.get('execution', {})
    if execution.get('engine', 'native') == 'native':
        python = resolve(config, execution.get('python'), follow_symlinks=False) or Path(sys.executable)
        probe = ('import importlib.metadata as m,json,sys; '
                 'wanted={"torch","torchvision","numpy","deepspeed","transformers","omegaconf","hydra-core"}; '
                 'print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"].lower():d.version '
                 'for d in m.distributions() if d.metadata["Name"].lower() in wanted}},sort_keys=True))')
        try:
            runtime = subprocess.check_output([str(python), '-c', probe], text=True, timeout=30).strip()
        except (OSError, subprocess.SubprocessError):
            runtime = 'unavailable:' + str(python)
    value = {'workspace': str(root), 'staged_inputs': staged, 'source': source,
             'recipe': resolved_recipe(config, stage), 'assets': assets,
             'execution': config.get('execution', {}), 'python_version': sys.version,
             'runtime': runtime,
             'image_sha256': sha256(image) if image and image.is_file() else None}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_qualification(config, stage):
    name = selected_recipe(config, stage)
    receipt = workspace_path(config) / 'logs' / ('verified-' + name + '.json')
    if not receipt.is_file():
        raise ValueError('Run the verify action in this workspace before training or pruning')
    value = read_json(receipt)
    if value.get('passed') is not True or value.get('fingerprint') != qualification_fingerprint(config, stage):
        raise ValueError('Verification is stale: source, image, input, configuration or workspace changed')


def command(config, stage, action='train'):
    name = selected_recipe(config, stage)
    cfg = resolved_recipe(config, stage)
    execution = config.get('execution', {})
    engine = execution.get('engine', 'native')
    image = resolve(config, execution.get('image'))
    if engine != 'native' and image is None:
        raise ValueError('Set execution.image to a training Apptainer image')
    modules = {
        'learn-masks': {'prepare': 'adaptive_pruning.prepare', 'verify': 'qualification', 'train': 'adaptive_pruning.train'},
        'prune': {'select': 'adaptive_pruning.select', 'export': 'adaptive_pruning.export', 'verify': 'adaptive_pruning.verify_export'},
        'recover': {'verify': 'qualification', 'train': 'recovery.train'},
    }
    if action not in modules[stage]:
        raise ValueError('Unsupported action ' + action + ' for ' + stage)
    package = Path(__file__).resolve().parent
    if engine == 'native':
        python = resolve(config, execution.get('python'), follow_symlinks=False) or Path(sys.executable)
        args = [str(python)]
    else:
        args = [engine, 'exec', '--nv', '--cleanenv', '--containall',
                '--bind', str(package) + ':/opt/bagel/src/bagel_robotics:ro',
                '--bind', str(workspace_path(config)) + ':/bagel/workspace:rw']
    for key, record in bindings(config, stage).items():
        host = record['host_path']
        if host is None:
            raise ValueError('Set paths.' + key + ' before generating a launch command')
        if any(char in str(host) for char in (':', ',', '\n')):
            raise ValueError('Asset paths may not contain colons, commas or newlines')
        if engine != 'native':
            args += ['--bind', str(host) + ':' + record['container_path'] + ':ro']
    env = {'PYTHONPATH': '/opt/bagel/src', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'VPP2_LOAD_META': '1', 'WANDB_MODE': 'disabled',
           'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID', 'REQUIRES_SLURM'),
           'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'REQUIRES_ALLOCATED_GPUS')}
    if engine != 'native':
        for key, value in env.items():
            args += ['--env', key + '=' + value]
        args += [str(image), 'python']
    single = action in {'prepare', 'verify'}
    if not single:
        args += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=' + str(cfg['train']['world_size'])]
    args += ['-m', 'bagel_robotics.' + modules[stage][action], '--config', cfg['root'] + '/config.json']
    if modules[stage][action] == 'qualification':
        args += ['--stage', name]
    return args


def run(config, stage, action):
    report = doctor(config, stage)
    if not report['passed']:
        raise ValueError('\n'.join(report['errors']))
    if action in {'train', 'select', 'export'}:
        require_qualification(config, stage)
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Training and GPU checks require a Slurm allocation')
    visible = [item for item in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if item]
    expected = resolved_recipe(config, stage)['train']['world_size']
    if not visible or (not (stage == 'prune' and action == 'verify') and len(visible) != expected):
        raise ValueError('Use the GPU allocation required by the selected recipe')
    args = command(config, stage, action)
    if not shutil.which(args[0]):
        raise ValueError('Training container engine is not installed: ' + args[0])
    image = resolve(config, config.get('execution', {}).get('image'))
    if config.get('execution', {}).get('engine', 'native') != 'native' and not image.is_file():
        raise ValueError('Training container image does not exist: ' + str(image))
    environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', VPP2_LOAD_META='1', WANDB_MODE='disabled')
    result = subprocess.run(args, env=environment, check=False).returncode
    if result == 0 and action == 'verify' and stage in {'learn-masks', 'recover'}:
        write_json(workspace_path(config) / 'logs' / ('verified-' + selected_recipe(config, stage) + '.json'),
                   {'passed': True, 'fingerprint': qualification_fingerprint(config, stage),
                    'scope': 'CPU and short GPU correctness checks; not a full experiment'})
    return result
