"""Immutable scientific choices. All block indices are zero based."""
import hashlib
import json
import math
from pathlib import Path

CANDIDATES = {
    "E1": (11, 15, 19, 24, 28, 32, 36, 39),
    "E2": (2, 5, 8, 11, 15, 19, 24, 39),
    "W1": (),
    "W2": (),
}
PRECISION_TASKS = (
    "build_tower", "deposit_coin", "fasten_screws", "insert_key",
    "insert_tubes", "play_Xylophone", "plug_in_charger", "pour_balls_into_vase",
)
ANCHORS = (0, 1, 4, 7, 9, 13, 17, 22, 26, 30, 34)
SCHEMA = "vpp2-compact-kv-v1"


def validate_architecture(indices, depth=40):
    values = tuple(indices)
    if any(type(i) is not int for i in values):
        raise ValueError("Layer indices must be integers, zero based")
    if values != tuple(sorted(set(values))) or any(i < 0 or i >= depth for i in values):
        raise ValueError("Layer indices must be sorted, unique and inside the native depth")
    return values


def architecture(candidate, selection=None):
    if candidate == 'learned-mask':
        from vpp2.width_kv.slicing import validate_indices
        if selection is None or selection.get('method') != 'learned_action_kv_masks':
            raise ValueError('Missing learned channel-selection manifest')
        validate_indices(selection['hidden_indices'],5120,2048)
        if len(selection['ffn_indices']) != 40: raise ValueError('Expected 40 FFN selections')
        for indices in selection['ffn_indices']: validate_indices(indices,13824,8192)
        return dict(schema='vpp2-compact-width-v1',candidate=candidate,native_depth=40,
                    kv_only_indices=[],hidden_dim=2048,ffn_dim=8192,num_heads=40,head_dim=128,
                    temporal_planes=5,tokens_per_plane=390,selection=selection,inference_teacher_required=False)
    if candidate not in CANDIDATES:
        raise ValueError(f"Expected E1, E2, W1 or W2, got {candidate}")
    if candidate in ('W1', 'W2'):
        from vpp2.width_kv.slicing import validate_indices
        if selection is not None:
            validate_indices(selection['hidden_indices'], 5120, 2048)
            if len(selection['ffn_indices']) != 40:
                raise ValueError('Expected all 40 FFN index sets')
            for indices in selection['ffn_indices']:
                validate_indices(indices, 13824, 8192)
            expected = 'evenly_spaced' if candidate == 'W1' else 'action_sensitivity'
            if selection.get('method') != expected or selection.get('candidate') != candidate:
                raise ValueError('Selection method/identity mismatch')
        return dict(schema='vpp2-compact-width-v1', candidate=candidate, native_depth=40,
                    kv_only_indices=[], hidden_dim=2048, ffn_dim=8192, num_heads=40, head_dim=128,
                    temporal_planes=5, tokens_per_plane=390, selection=selection,
                    inference_teacher_required=False)
    if selection is not None:
        raise ValueError('Legacy depth-only models have no width selection')
    return dict(schema=SCHEMA, candidate=candidate, native_depth=40,
                kv_only_indices=list(CANDIDATES[candidate]), hidden_dim=5120,
                num_heads=40, head_dim=128, temporal_planes=5,
                tokens_per_plane=390, inference_teacher_required=False)


def stable_seed(seed, *parts):
    payload = json.dumps([seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def lr_multiplier(update, total=1000, warmup=50, final=0.1):
    if not 0 <= update <= total or not 0 <= warmup < total:
        raise ValueError("Invalid learning-rate schedule position")
    if update < warmup:
        return (update + 1) / warmup
    phase = (update - warmup) / (total - warmup)
    return final + (1 - final) * (1 + math.cos(math.pi * phase)) / 2


def kv_weight(update, maximum=0.1, ramp=50):
    return maximum * min(1.0, max(0.0, update / ramp))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()
