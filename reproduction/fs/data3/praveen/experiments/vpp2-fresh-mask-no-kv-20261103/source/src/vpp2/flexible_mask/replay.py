"""Replay the reference run's regularizer strengths for the no-K/V ablation."""
import math
from .coefficient import Coefficient


def validate(config):
    r=config.get('regularization_replay')
    if r is None:return
    required={'schema','reference_root','reference_config_sha256','resolved_update','base_coefficient',
              'pressure_start','pressure_ramp','pressure_multiplier','final_coefficient'}
    if set(r)!=required or r['schema']!='reference-regularization-replay-v1':
        raise ValueError('Invalid regularization replay')
    if 'resume_from' in config or 'pressure_continuation' in config:
        raise ValueError('Fresh ablation cannot resume the previous mask/video state')
    if config['train']['kv_weight']!=0.:
        raise ValueError('This replay is the explicit no-K/V ablation')
    if (r['resolved_update'],r['pressure_start'],r['pressure_ramp'],r['pressure_multiplier'])!=(200,1251,250,2.):
        raise ValueError('Reference regularizer schedule changed')
    if not math.isfinite(r['base_coefficient']) or not 0<r['base_coefficient']<=config['size_coefficient']['upper_cap']/2:
        raise ValueError('Invalid reference size coefficient')
    if not math.isclose(r['final_coefficient'],2*r['base_coefficient'],rel_tol=1e-12):
        raise ValueError('Reference final size penalty changed')


def initial_coefficient(config):
    r=config.get('regularization_replay')
    if r is None:return Coefficient(config['size_coefficient']['upper_cap'])
    validate(config)
    # This value was measured in the reference run; do not claim new calibration.
    return Coefficient(cap=config['size_coefficient']['upper_cap'],measured=r['base_coefficient'],
                       applied=r['base_coefficient'],resolved_update=r['resolved_update'])


def weight(coefficient,config,update):
    validate(config);r=config['regularization_replay']
    if coefficient.unresolved or coefficient.applied!=r['base_coefficient'] or coefficient.resolved_update!=r['resolved_update']:
        raise ValueError('Regularizer replay coefficient state changed')
    progress=min(max((update-r['pressure_start'])/r['pressure_ramp'],0.),1.)
    return min(coefficient.weight(update)*(1+progress*(r['pressure_multiplier']-1)),r['final_coefficient'])
