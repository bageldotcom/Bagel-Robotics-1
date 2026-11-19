"""Average per-window diagnostic gradients without retaining a large batch graph."""
import torch


def measure_in_chunks(rows, positions, chunk_size, measure):
    if len(rows)!=len(positions) or not rows or type(chunk_size) is not int or chunk_size<1:
        raise ValueError('Invalid calibration batch')
    totals=None
    for start in range(0,len(rows),chunk_size):
        stop=min(start+chunk_size,len(rows))
        gradients=measure(rows[start:stop],positions[start:stop])
        if set(gradients)!={'gt','video','kv'} or any(len(v)!=3 for v in gradients.values()):
            raise ValueError('Calibration objective/family inventory changed')
        if any(g.requires_grad or g.dtype!=torch.float32 for values in gradients.values() for g in values):
            raise ValueError('Expected detached FP32 mask gradients')
        if totals is None:totals={k:[torch.zeros_like(g) for g in values] for k,values in gradients.items()}
        for key,values in gradients.items():
            for total,g in zip(totals[key],values):
                if total.shape!=g.shape:raise ValueError('Calibration gradient shape changed')
                total.add_(g,alpha=(stop-start)/len(rows))
        del gradients
    return totals


def release_unused_cache(device):
    """Release allocator-owned free blocks for NCCL, preserving all live tensors."""
    device=torch.device(device)
    if device.type!='cuda':return dict(released_gib=0.)
    allocated_before=torch.cuda.memory_allocated(device)
    reserved_before=torch.cuda.memory_reserved(device)
    free_before,total=torch.cuda.mem_get_info(device)
    torch.cuda.empty_cache()
    allocated_after=torch.cuda.memory_allocated(device)
    reserved_after=torch.cuda.memory_reserved(device)
    free_after,_=torch.cuda.mem_get_info(device)
    return dict(released_gib=max(reserved_before-reserved_after,0)/2**30,
        allocated_before_gib=allocated_before/2**30,allocated_after_gib=allocated_after/2**30,
        reserved_before_gib=reserved_before/2**30,reserved_after_gib=reserved_after/2**30,
        device_free_before_gib=free_before/2**30,device_free_after_gib=free_after/2**30)
