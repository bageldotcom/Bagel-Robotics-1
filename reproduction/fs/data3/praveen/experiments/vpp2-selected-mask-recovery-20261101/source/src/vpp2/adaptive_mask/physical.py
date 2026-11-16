"""Audit the actual compact policy against cached, immutable native predictions."""
import torch
from omegaconf import OmegaConf
from vpp2.compact_kv.model import native_config
from .export import build_compact,sliced_state
from .masking import install
from .monitor import native_evaluate


class OpenMasks:
    def forward_gates(self,mode='binary'):
        if mode!='binary':raise ValueError('Physical audit is inference only')
        return None,None,None


def require_original_predictions(model,entries,rank,world):
    state=getattr(model,'_adaptive_native_monitor',None)
    expected={r['id'] for i,r in enumerate(entries) if r.get('owner_index',i)%world==rank}
    if state is None or not expected.issubset(state['teacher']):
        raise ValueError('Physical audit requires original native predictions for every local window')
    return {key:state['teacher'][key].clone() for key in expected}


@torch.no_grad()
def native_compact_audit(config,panel,model,controller,inventory,rank,world,entries,masked_native):
    saved=require_original_predictions(model,entries,rank,world)
    selection=controller.selection();original=model.video_expert
    original.cpu();torch.cuda.empty_cache()
    args=OmegaConf.to_container(native_config(config).model.video_dit_config,resolve=True)
    compact=build_compact(args,selection,controller.geometry)
    state,_=sliced_state(original.state_dict(),compact,selection,controller.geometry)
    compact.load_state_dict(state,strict=True,assign=True);del state
    count=sum(p.numel() for p in compact.parameters())
    if count!=inventory.count(selection):raise ValueError('Physical audit parameter count mismatch')
    compact=compact.to(device=model.device,dtype=torch.bfloat16)
    branch=install(compact).eval()
    # native_evaluate reuses the original teacher cache, while its student path
    # executes the physically smaller network with no mask modules or teacher KV.
    result=native_evaluate(config,panel,model,branch,OpenMasks(),rank,world,
                           parent=masked_native,entries_override=entries)
    for key,value in saved.items():
        if not torch.equal(value,model._adaptive_native_monitor['teacher'][key]):
            raise AssertionError('Physical audit altered original native predictions')
    result.update(evaluated_model='physically_compact',physical_video_parameters=count,
                  original_predictions_reused=True,masking_required=False)
    return result
