"""Motion and ordered-history descriptors for demonstration-window selection."""
from pathlib import Path
import numpy as np
from bagel_robotics.data_selection.manifest import digest,file_hash
HORIZON = 32
MOTION_NAMES = [
    'left_path', 'right_path', 'left_rotation', 'right_rotation',
    'left_grip_variation', 'right_grip_variation', 'left_displacement', 'right_displacement',
    'left_active_fraction', 'right_active_fraction', 'both_active_fraction',
    'relative_hand_distance_change', 'left_turning', 'right_turning',
    'left_slow_rotation', 'right_slow_rotation',
    'left_grip_mean', 'right_grip_mean',
    'left_x', 'left_y', 'left_z', 'right_x', 'right_y', 'right_z',
    'left_initial_grip', 'right_initial_grip',
    'left_state_action_gap', 'right_state_action_gap',
]
GROUPS = {'translation': list(range(0, 2)) + list(range(6, 8)) + list(range(12, 14)),
          'rotation': [2, 3, 14, 15], 'grip': [4, 5, 16, 17, 24, 25],
          'two_arm': [8, 9, 10, 11], 'state': list(range(18, 24)) + [26, 27]}

def history_indices(t):
    return [0] + [max(0, int(t) + o) for o in range(-175, 1, 25)]

def candidate_starts(length, stride=8):
    if length < HORIZON:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.r_[np.arange(0, length - HORIZON + 1, stride), length - HORIZON]).astype(np.int64)

def valid_actions(a):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 16 or not np.isfinite(a).all():
        raise ValueError('Require finite raw [T,16] EE actions/states')
    for start in (3, 11):
        if np.any(np.linalg.norm(a[:, start:start+4], axis=-1) < 1e-8):
            raise ValueError('Zero-norm quaternion')
    return a

def quaternion_steps(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.r_[0., 2 * np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(-1)), 0, 1))]

def motion_descriptors(actions, states, stride=8):
    a = valid_actions(actions).astype(np.float64)
    s = valid_actions(states).astype(np.float64)
    if a.shape != s.shape:
        raise ValueError('Action/state length mismatch')
    xyz = a[:, [0,1,2,8,9,10]].reshape(-1,2,3)
    dp = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=-1)
    rot = np.stack([quaternion_steps(a[:,3:7]), quaternion_steps(a[:,11:15])], -1)
    grip = np.abs(np.diff(a[:,[7,15]], axis=0, prepend=a[:1,[7,15]]))
    active = (dp > .0005) | (rot > np.deg2rad(.5)) | (grip > .02)
    starts = candidate_starts(len(a), stride)
    # Add event-centred candidates; these are command transitions, not contacts.
    event = np.flatnonzero((grip.max(-1) > .1) | (rot.max(-1) > np.quantile(rot.max(-1), .95)))
    if len(event):
        starts = np.unique(np.r_[starts, np.clip(event-8, 0, len(a)-32), np.clip(event-16, 0, len(a)-32)])
    vectors = []
    for t in starts:
        sl = slice(t,t+32)
        path = dp[sl].sum(0); angle = rot[sl].sum(0)
        displacement = np.linalg.norm(xyz[t+31]-xyz[t], axis=-1)
        turning = np.clip(1-displacement/np.maximum(path,1e-9), 0, 1)
        relative = abs(np.linalg.norm(xyz[t+31,0]-xyz[t+31,1])-np.linalg.norm(xyz[t,0]-xyz[t,1]))
        slow_angle = (rot[sl] * (dp[sl] <= np.quantile(dp, .5, axis=0))).sum(0)
        gap = np.linalg.norm((a[t,[0,1,2,8,9,10]]-s[t,[0,1,2,8,9,10]]).reshape(2,3), axis=-1)
        vectors.append(np.r_[path, angle, grip[sl].sum(0), displacement,
                             active[sl].mean(0), active[sl].all(-1).mean(), relative,
                             turning, slow_angle, a[sl][:,[7,15]].mean(0),
                             s[t,[0,1,2,8,9,10]], a[t,[7,15]], gap])
    return starts.astype(np.int64), np.asarray(vectors, dtype=np.float64)

def robust_transform(x):
    median = np.median(x, axis=0)
    scale = np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
    fallback = np.quantile(x,.95,axis=0)-np.quantile(x,.05,axis=0)
    scale = np.where(scale>1e-6, scale, np.where(fallback>1e-6, fallback, 1.))
    return median, scale

def transform(x, median, scale):
    return np.clip((x-median)/scale,-5,5)

def fingerprint_window(config,row):
    base=Path(config['demonstrations'])/row['episode_path']
    a=np.load(base/'actions.npy',mmap_mode='r');s=np.load(base/'states.npy',mmap_mode='r');f=np.load(base/'frames.npy',mmap_mode='r')
    h=hashlib.sha256();h.update(row['prompt'].encode());h.update(np.asarray(s[row['frame']]).tobytes())
    h.update(np.asarray(a[row['frame']:row['frame']+32]).tobytes())
    for t in row['history_indices']:h.update(np.asarray(f[t]).tobytes())
    return h.hexdigest()
