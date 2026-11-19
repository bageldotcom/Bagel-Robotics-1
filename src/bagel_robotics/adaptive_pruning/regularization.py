"""Size-cost calibration and explicit continuous regularizer schedules."""
import math
from bagel_robotics._internal.training.coefficient import Coefficient


def initial_coefficient(config):
    settings=config['size_coefficient'];value=settings.get('initial_value')
    if value is None:return Coefficient(cap=settings['upper_cap'])
    return Coefficient(cap=settings['upper_cap'],measured=value,applied=value,
                       resolved_update=settings['resolved_update'])


def size_weight(coefficient,config,update):
    settings=config['size_coefficient'];base=coefficient.weight(update)
    multiplier=settings.get('pressure_multiplier',1.)
    progress=min(max((update-settings.get('pressure_start',update))/settings.get('pressure_ramp',1),0.),1.)
    return min(base*(1+progress*(multiplier-1)),
               coefficient.applied*multiplier if coefficient.applied is not None else 0.)


def learning_rate_fraction(update,total,warmup,final=.1):
    if update<=warmup:return update/warmup
    return final+(1-final)*.5*(1+math.cos(math.pi*min((update-warmup)/(total-warmup),1)))


def recovery_learning_rate(config,update):
    settings=config['recovery']
    if update<=settings['hold_updates']:return settings['learning_rate']
    progress=(update-settings['hold_updates'])/(settings['schedule_updates']-settings['hold_updates'])
    progress=min(progress,1.)
    return settings['learning_rate']*(settings['final_lr_fraction']+(1-settings['final_lr_fraction'])*.5*(1+math.cos(math.pi*progress)))
