"""Explicit training exclusions with unchanged full-Precision held-out data."""
import csv
import io
import json
import math
from pathlib import Path

from vpp2.compact_kv.spec import PRECISION_TASKS, atomic_json, file_sha


def task_scope(config):
    scope = config['task_scope']
    train = tuple(scope['training_tasks'])
    excluded = tuple(scope['excluded_training_tasks'])
    validation = tuple(scope['validation_tasks'])
    for group in (train, excluded, validation):
        if len(group) != len(set(group)):
            raise ValueError('Duplicate task in the experiment scope')
    if not train or set(train) & set(excluded):
        raise ValueError('Training and excluded tasks must be disjoint and training nonempty')
    if set(train) | set(excluded) != set(PRECISION_TASKS) or set(validation) != set(PRECISION_TASKS):
        raise ValueError('Training/excluded tasks must partition all eight validation tasks')
    if scope['checkpoint_selection'] != 'training_tasks_equal_weight':
        raise ValueError('Checkpoint selection must use the retained tasks, equally weighted')
    return train, excluded, validation


def read_csv(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def csv_bytes(fields, rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def immutable_bytes(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError('Refusing to overwrite a different data artifact: ' + str(path))
        return
    temporary = path.with_name(path.name + '.partial')
    temporary.write_bytes(data)
    temporary.replace(path)


def prepare_filtered_data(config):
    """Copy small manifests only; reuse immutable media, embeddings and native stats."""
    train_tasks, excluded, validation_tasks = task_scope(config)
    source = (Path(config['previous_root']) / 'data').resolve()
    data = Path(config['root']) / 'data'
    if data.is_symlink():
        raise RuntimeError('Filtered data must not alias the predecessor data directory')
    prepared = data / 'prepared'
    prepared.mkdir(parents=True, exist_ok=True)
    original_ready = json.loads((source / 'ready.json').read_text())
    # Validate all small upstream artifacts, including the unchanged normalization
    # and critical-phase index, before making a filtered view of the data.
    for name, digest in original_ready['files'].items():
        if file_sha(source / 'prepared' / name) != digest:
            raise RuntimeError('Predecessor data artifact changed: ' + name)
    fields, original = read_csv(source / 'prepared/train.csv')
    _, val = read_csv(source / 'prepared/val.csv')
    if len(original) != 792 or len(val) != 8:
        raise RuntimeError('Unexpected original Precision split')
    if set(r['task_name'] for r in original) != set(PRECISION_TASKS):
        raise RuntimeError('Original training task set changed')
    if set(r['task_name'] for r in val) != set(validation_tasks):
        raise RuntimeError('All eight held-out tasks must be retained')
    episode_id = lambda r: (r['task_name'], int(r['episode_index']))
    if len({episode_id(r) for r in original}) != len(original) or len({episode_id(r) for r in val}) != len(val):
        raise RuntimeError('Duplicate episode identities')
    if {episode_id(r) for r in original} & {episode_id(r) for r in val}:
        raise RuntimeError('Training/validation episode overlap')
    selected = [r for r in original if r['task_name'] in train_tasks]
    removed = [r for r in original if r['task_name'] in excluded]
    counts = {task: sum(r['task_name'] == task for r in original) for task in PRECISION_TASKS}
    if set(counts.values()) != {99}:
        raise RuntimeError('Expected 99 training episodes per task')
    immutable_bytes(prepared / 'train.csv', csv_bytes(fields, selected))
    immutable_bytes(prepared / 'val.csv', (source / 'prepared/val.csv').read_bytes())
    immutable_bytes(prepared / 'all.csv', csv_bytes(fields, selected + val))
    immutable_bytes(data / 'excluded_train.csv', csv_bytes(fields, removed))
    for name in ('critical_index', 'dataset_stats.json', 'prompts.json', 'text_embeds'):
        target = (source / 'prepared' / name).resolve()
        destination = prepared / name
        if not destination.exists():
            destination.symlink_to(target, target_is_directory=target.is_dir())
        if destination.resolve() != target:
            raise RuntimeError('Unexpected shared prepared artifact: ' + name)
    media = (source / 'media').resolve()
    if not (data / 'media').exists():
        (data / 'media').symlink_to(media, target_is_directory=True)
    if (data / 'media').resolve() != media:
        raise RuntimeError('Unexpected shared media source')
    immutable_bytes(data / 'precision_holdout.json', (source / 'precision_holdout.json').read_bytes())
    windows = lambda rows: sum(int(r['trim_end']) - int(r['trim_start']) for r in rows)
    record = dict(schema='learned-width-filtered-precision-data-v1', source=str(source),
        source_ready_sha256=file_sha(source / 'ready.json'), task_scope=config['task_scope'],
        train=len(selected), val=len(val), excluded_train=len(removed),
        train_windows=windows(selected), excluded_train_windows=windows(removed),
        validation_windows=128, training_episodes_by_task={t: counts[t] for t in train_tasks},
        training_windows_by_task={t: windows([r for r in selected if r['task_name'] == t]) for t in train_tasks},
        teacher_stats_sha256=original_ready['teacher_stats_sha256'],
        validation_sha256=file_sha(source / 'prepared/val.csv'),
        source_train_sha256=file_sha(source / 'prepared/train.csv'),
        initialization_note='W2 channel-score initialization was derived from the original eight-task training split; no validation data. New mask learning, recovery and reliability calibration use six tasks only.')
    record['files'] = {name: file_sha(prepared / name) for name in original_ready['files'] if name != 'preparation.json'}
    record['excluded_train_sha256'] = file_sha(data / 'excluded_train.csv')
    # preparation.json describes this filtered view; no upstream file is edited.
    preparation = dict(train=len(selected), val=len(val), prompts=8, media_root=str(media),
        statistics_source='unchanged original teacher full-dataset statistics', task_scope=config['task_scope'])
    immutable_bytes(prepared / 'preparation.json', (json.dumps(preparation, indent=2, sort_keys=True) + '\n').encode())
    record['files']['preparation.json'] = file_sha(prepared / 'preparation.json')
    if (data / 'ready.json').exists() and json.loads((data / 'ready.json').read_text()) != record:
        raise RuntimeError('Filtered data provenance changed')
    atomic_json(data / 'ready.json', record)
    return record


def validate_training_dataset(dataset, config):
    train_tasks, _, _ = task_scope(config)
    record = json.loads((Path(config['root']) / 'data/ready.json').read_text())
    if set(dataset.metadata.task_name) != set(train_tasks):
        raise RuntimeError('Excluded or missing training task reached the runtime dataset')
    if len(dataset.metadata) != record['train'] or len(dataset) != record['train_windows']:
        raise RuntimeError('Runtime training episode/window counts differ from the pinned manifest')
    return record


def aggregate_validation(rows, config):
    """Equal-task reports; unavailable reliability metrics stay null, never zero."""
    def check_finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError('Nonfinite validation result cannot be reported or selected')
        if isinstance(value, dict):
            for item in value.values(): check_finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value: check_finite(item)
    check_finite(rows)
    train, excluded, validation = task_scope(config)
    tasks = set(row['task'] for row in rows)
    if tasks != set(validation):
        raise RuntimeError('Incomplete all-eight-task validation')
    ignored = {'item_id', 'trajectory', 'step'}
    keys = sorted({k for row in rows for k, value in row.items()
                   if k not in ignored and isinstance(value, float)})
    vector_keys = sorted({k for row in rows for k in row if k.endswith('_mse_by_action_dimension')})
    def mean(group):
        result = {key: (sum(row[key] for row in group) / len(group)
                       if all(isinstance(row.get(key), float) for row in group) else None)
                  for key in keys}
        for key in vector_keys:
            if not all(isinstance(row.get(key),list) and len(row[key])==16 for row in group):
                raise ValueError('Incomplete per-dimension action diagnostics')
            result[key] = [sum(row[key][d] for row in group)/len(group) for d in range(16)]
        return result
    by_task = {task: mean([row for row in rows if row['task'] == task]) for task in validation}
    by_scope = {name: mean([by_task[task] for task in members])
                for name, members in [('primary', train), ('excluded_diagnostic', excluded), ('full_precision', validation)]
                if members}
    return dict(by_task=by_task, by_scope=by_scope, metric_scope='full_precision',
                checkpoint_selection_scope='primary', **by_scope['full_precision'])


def select_checkpoints(rows, config):
    training, _, validation = task_scope(config)
    if not rows:
        raise RuntimeError('No validated checkpoints to select')
    def score(row, key):
        if not row['full'] or row['windows'] != 128 or set(row['by_task']) != set(validation):
            raise RuntimeError('Checkpoint requires complete all-eight-task offline validation')
        return sum(row['by_task'][task][key] for task in training) / len(training)
    return (min(rows, key=lambda row: score(row, 'student_chunk_mse')),
            min(rows, key=lambda row: score(row, 'chunk_teacher_mse')))
