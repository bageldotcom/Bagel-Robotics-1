"""Explicit acceptance policy; video diagnostics and training objectives stay intact."""
LEGACY = 'action_and_video_v1'
ACTION_ONLY = 'action_only_v1'


def policy_mode(config):
    policy = config.get('quality_gate', {'mode': LEGACY})
    if not isinstance(policy, dict) or set(policy) != {'mode'} or policy['mode'] not in (LEGACY, ACTION_ONLY):
        raise ValueError('Unknown or malformed quality acceptance policy')
    return policy['mode']


def quality_flags(reports):
    reports = list(reports)
    if not reports:
        raise ValueError('No quality measurements supplied')
    return dict(passed=all(x['passed'] for x in reports),
                clearly_passing=all(x['clearly_passing'] for x in reports),
                resolved_violation=any(x['resolved_violation'] for x in reports))


def monitor_acceptance(config, action, video, components):
    # An average action score cannot override a failing position/orientation/gripper check.
    if set(components) != {'position', 'quaternion_coordinates', 'gripper'}:
        raise ValueError('All native action components are required for acceptance')
    mode = policy_mode(config)
    action_flags = quality_flags([action, *components.values()])
    joint_flags = quality_flags([action, video, *components.values()])
    return dict(quality_gate_mode=mode, video_is_acceptance_gate=mode == LEGACY,
                action_acceptance=action_flags, action_video_diagnostic=joint_flags,
                **(action_flags if mode == ACTION_ONLY else joint_flags))
