"""Keep diagnostics separate from training state and handle phase boundaries."""
from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def preserve_rng():
    """Restore CPU/Python/NumPy and this rank's initialized CUDA RNG, even on error."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def interruption_needed(requested, update, total):
    """A stop arriving at the final update must complete, not requeue a finished phase."""
    if not 0 <= update <= total:
        raise ValueError('Invalid phase update')
    return bool(requested and update < total)
