import copy
from types import SimpleNamespace
import pytest
import torch
from vpp2.adaptive_mask.masking import install,ExactOpenLayerNorm,MaskedRMSNorm
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT
from vpp2.flexible_mask.data import prepare_cached
from vpp2.adaptive_mask import data as native_data


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('mode',['soft','binary','open'])
def test_cached_open_flags_preserve_complete_branch_and_all_gradients(dtype,mode):
    torch.manual_seed(24)
    args=dict(hidden_dim=80,ffn_dim=128,num_layers=3,num_heads=4,attn_head_dim=8,
              freq_dim=8,text_dim=10,in_dim=8,out_dim=2,cross_attn_norm=True)
    a=install(WanVideoDiT(**args).to(dtype)).train()
    b=copy.deepcopy(a);b.video.cache_mask_open_flags=True
    for model in (a,b):model.video.use_gradient_checkpointing=True
    inputs=dict(context=torch.randn(1,3,10).to(dtype),context_mask=torch.ones(1,3,dtype=torch.bool),
                clip_fea=torch.randn(1,257,1280).to(dtype),condition_latents=torch.randn(1,2,1,4,4).to(dtype))
    noise=torch.randn(1,2,2,4,4).to(dtype);time=torch.tensor([1000.]);attention=torch.ones(8,8,dtype=torch.bool)
    masks=[]
    for shape in ((80,),(2,128),(2,4)):
        x=torch.rand(shape)*.8+.1
        if mode=='open':x.fill_(1)
        elif mode=='binary':x=(x>.3).float();x[...,0]=1
        masks.append(x.requires_grad_())
    other=[x.detach().clone().requires_grad_() for x in masks]
    outputs=[]
    for model,gates in ((a,masks),(b,other)):
        output=model(inputs,noise,time,attention,*gates)
        objective=output['video'].float().square().mean()+sum(c[k].float().square().mean() for c in output['cache'] for k in ('k','v'))
        objective.backward();outputs.append(output)
    assert torch.equal(outputs[0]['video'],outputs[1]['video'])
    for x,y in zip(outputs[0]['cache'],outputs[1]['cache']):
        for k in ('k','v'):assert torch.equal(x[k],y[k])
    for x,y in zip(masks,other):assert torch.equal(x.grad,y.grad)
    for (na,x),(nb,y) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb
        assert (x.grad is None)==(y.grad is None)
        if x.grad is not None:assert torch.equal(x.grad,y.grad),na


@pytest.mark.parametrize('initially_cached',[set(),{1},{1,2}])
def test_batch_cache_verifies_each_successful_payload_once_and_restores_reader(monkeypatch,initially_cached):
    stored={i:{'value':i*7} for i in initially_cached};reads=[]
    def load(key):
        if key in stored:reads.append(key)
        return stored.get(key)
    cache=SimpleNamespace(load=load);reader=SimpleNamespace(runtime_cache=cache,reuse_prepared_payloads=True)
    def native(model,reader,rows,positions,seed,strata):
        values=[]
        for row in rows:
            value=reader.runtime_cache.load(row['id'])
            if value is None:
                assert len(rows)==1
                value={'value':row['id']*7};stored[row['id']]=value
            values.append(value['value'])
        return values,positions,strata
    monkeypatch.setattr(native_data,'prepare',native)
    assert prepare_cached(None,reader,[{'id':1},{'id':2}],[9,10],100008,[0,1])==([7,14],[9,10],[0,1])
    assert reader.runtime_cache is cache
    assert sorted(reads)==[1,2]


def test_batch_cache_exception_restores_original_reader(monkeypatch):
    cache=SimpleNamespace(load=lambda key:None)
    reader=SimpleNamespace(runtime_cache=cache,reuse_prepared_payloads=True)
    def broken(*args):raise RuntimeError('injected')
    monkeypatch.setattr(native_data,'prepare',broken)
    with pytest.raises(RuntimeError,match='injected'):
        prepare_cached(None,reader,[{'id':1},{'id':2}],[0,1],1,[0,0])
    assert reader.runtime_cache is cache


def test_chained_cache_origin_validates_each_parent(tmp_path,monkeypatch):
    import json
    from vpp2.flexible_mask.data import reader_configuration
    from vpp2.flexible_mask import continuation
    original={'root':str(tmp_path/'original'),'marker':'original-cache'}
    middle={'root':str(tmp_path/'middle'),'resume_from':{'root':original['root']}}
    current={'root':str(tmp_path/'current'),'resume_from':{'root':middle['root']}}
    from pathlib import Path
    for c in (original,middle):
        p=Path(c['root']);p.mkdir();(p/'config.json').write_text(json.dumps(c))
    calls=[]
    def validate(c,stage):
        calls.append(c['root']);assert stage=='mask'
        return Path(c['resume_from']['root']),{}
    monkeypatch.setattr(continuation,'validate_parent',validate)
    assert reader_configuration(current)==original
    assert calls==[current['root'],middle['root']]


def test_large_batch_is_candidate_only_and_execution_options_are_strict():
    import json
    from pathlib import Path
    from vpp2.flexible_mask.config import validate
    from vpp2.flexible_mask.continuation import scientific_recipe
    c=json.loads((Path(__file__).resolve().parents[2]/'configs/flexible_mask/full30k.json').read_text())
    original=copy.deepcopy(c);c['train']['micro_batch']=4
    c['execution']={'cached_norm_flags':True,'skip_nograd_checkpointing':True,
                    'mixed_attention_checkpointing':False,'deduplicate_conditioning_reads':True}
    validate(c);assert scientific_recipe(c)==scientific_recipe(original)
    c['execution']['cached_norm_flags']='true'
    with pytest.raises(ValueError,match='boolean'):validate(c)
