"""Read-only native mismatch diagnosis; never writes a readiness pass receipt."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from vpp2.adaptive_mask import preflight, masking
from vpp2.adaptive_mask.runtime import forward_video, predict


def gather_layer_norm(self, x, gate=None):
    # Preserve the existing analytic mask derivative, but evaluate retained
    # coordinates with the exact normalization kernel/shape used at export.
    result = super(masking.ExactOpenLayerNorm, self).forward(x, gate)
    if gate is None:
        return result
    ids = gate.detach().nonzero().flatten()
    kept = x.index_select(-1, ids)
    weight = None if self.weight is None else self.weight.index_select(0, ids)
    bias = None if self.bias is None else self.bias.index_select(0, ids)
    if self.native_wan_norm:
        kept = F.layer_norm(kept.float(), (len(ids),), None if weight is None else weight.float(),
                            None if bias is None else bias.float(), self.eps).to(x.dtype)
    else:
        kept = F.layer_norm(kept, (len(ids),), weight, bias, self.eps)
    exact = result.detach().index_copy(-1, ids, kept.detach())
    return exact + (result - result.detach())


def gather_rms_norm(self, x, gate=None):
    if gate is None:
        return super(masking.MaskedRMSNorm, self).forward(x)
    value = x.float(); g = gate.float(); mass = g.sum()
    value = value * torch.rsqrt((value.square() * g).sum(-1, keepdim=True) / mass + self.eps)
    result = value.to(x.dtype) * self.weight
    ids = gate.detach().nonzero().flatten()
    kept = x.index_select(-1, ids).float()
    kept = (kept * torch.rsqrt(kept.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)
    kept = kept * self.weight.index_select(0, ids)
    return result.detach().index_copy(-1, ids, kept.detach()) + (result - result.detach())


def compare(cache, video, action, expected_cache, expected_video, expected_action):
    rows=[]
    for i,(a,b) in enumerate(zip(cache,expected_cache)):
        for k in ('k','v'):
            rows.append(dict(tensor=f'layer_{i:02d}_{k}',**preflight.error_report(a[k].cpu(),b[k].cpu())))
    for name,a,b in [('video',video,expected_video),('action',action,expected_action)]:
        rows.append(dict(tensor=name,**preflight.error_report(a.cpu(),b.cpu())))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());panel=json.loads(Path(c['panel_file']).read_text())
    try:
        preflight.native(c,panel)
    except AssertionError as error:
        if 'Native BF16 binary/export mismatch' not in str(error):
            raise
        tb=error.__traceback__
        while tb and tb.tb_frame.f_code.co_name!='native':tb=tb.tb_next
        if tb is None:raise
        f=tb.tb_frame.f_locals
        actual=f['actual']
        baseline=compare(actual['cache'],actual['video'],f['actual_action'],
                         f['expected_cache'],f['expected_video'],f['expected_action'])
        compact_cache=[{k:v.cpu() for k,v in layer.items()} for layer in actual['cache']]
        compact_video=actual['video'].cpu();compact_action=f['actual_action'].cpu()
        actual['cache']=compact_cache;actual['video']=compact_video
        f['compact'].cpu();torch.cuda.empty_cache()
        masking.ExactOpenLayerNorm.forward=gather_layer_norm
        masking.MaskedRMSNorm.forward=gather_rms_norm
        f['model'].video_expert.cuda();torch.cuda.empty_cache()
        with torch.no_grad():
            fixed=forward_video(f['branch'],f['parts'],f['controller'].forward_gates(mode='binary'))
            action=predict(f['model'],f['parts'],fixed['cache'])
        corrected=compare(fixed['cache'],fixed['video'],action,compact_cache,compact_video,compact_action)
        report=dict(diagnostic_only=True,baseline=baseline,selected_coordinate_norms=corrected,
                    baseline_max=max(r['relative_rmse'] for r in baseline),
                    selected_coordinate_norms_max=max(r['relative_rmse'] for r in corrected),
                    readiness_pass_written=False)
        path=Path(c['root'])/'checks/rounding_diagnostic.json';path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','selected_coordinate_norms')}),flush=True)
        print(json.dumps(dict(baseline_worst=sorted(baseline,key=lambda r:r['relative_rmse'],reverse=True)[:8],
                              corrected_worst=sorted(corrected,key=lambda r:r['relative_rmse'],reverse=True)[:8])),flush=True)
    else:
        raise RuntimeError('Previously failing native comparison unexpectedly passed; inspect reproducibility')


if __name__=='__main__':main()
