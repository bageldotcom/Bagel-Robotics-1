"""Pinned deterministic execution for width pilots, including attention backward."""
import os
import torch


def configure_deterministic_training():
    workspace = ':4096:8'
    current = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if current not in (None, workspace):
        raise RuntimeError('Width pilots require CUBLAS_WORKSPACE_CONFIG=' + workspace)
    if current is None and torch.cuda.is_initialized():
        raise RuntimeError('Configure deterministic width execution before CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return dict(schema='width-deterministic-execution-v1',
                deterministic_algorithms=True, warn_only=False,
                cudnn_benchmark=False, cudnn_deterministic=True,
                matmul_allow_tf32=False, cublas_workspace_config=workspace)
