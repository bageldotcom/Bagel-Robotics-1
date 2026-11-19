"""Execution-only recomputation settings; no objective or sampling changes."""


def configure_recomputation(model, config):
    settings = config.get('execution', {})
    count = len(model.video_expert.blocks)
    layers = settings.get('video_checkpoint_layers', list(range(count)))
    action = settings.get('action_checkpointing', True)
    if not isinstance(action, bool):
        raise ValueError('Action checkpointing must be boolean')
    if not isinstance(layers, list) or any(type(i) is not int or not 0 <= i < count for i in layers):
        raise ValueError('Invalid video checkpoint layer indices')
    if layers != sorted(set(layers)):
        raise ValueError('Video checkpoint layers must be sorted and unique')
    model.video_expert.use_gradient_checkpointing = bool(layers)
    model.video_expert.activation_checkpoint_layers = frozenset(layers)
    model.action_expert.use_gradient_checkpointing = action
    if 'mixed_attention_checkpointing' in settings:
        mixed=settings['mixed_attention_checkpointing']
        if not isinstance(mixed,bool):raise ValueError('Mixed-attention checkpointing must be boolean')
        model.mot.mot_checkpoint_mixed_attn=mixed


def checkpoint_layer(video, index):
    layers = getattr(video, 'activation_checkpoint_layers', None)
    return video.use_gradient_checkpointing and (layers is None or index in layers)
