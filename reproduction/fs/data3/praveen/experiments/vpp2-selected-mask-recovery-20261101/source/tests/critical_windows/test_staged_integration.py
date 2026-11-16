import json
import numpy as np
import pytest

from vpp2.critical_windows.selection import repair_candidate_grid,choose
from vpp2.critical_windows.prepare_staged import runtime_config


def test_sparse_candidate_repair_preserves_required_frames_without_descriptor_ranking():
    rows=[dict(episode=1,frame=t,length=89,role='selection_initial',temporal_quarter=t//24,
               relative_time=t/57,proxy_tags=[]) for t in (0,32,57)]
    anchors={1:dict(role='fit',frames=[48,80])}
    x=np.arange(3,dtype=np.float32)[:,None]
    augmented,features,plans,repairs=repair_candidate_grid(rows,x,2,{},anchors)
    assert len(repairs)==1
    assert not repairs[0]['invented_visual_descriptors']
    ids,reasons=choose(augmented,features,2,7,2,{},base=0,random_fraction=0,
                       observed_anchors=anchors,preplanned_windows=plans)
    starts=sorted(augmented[i]['frame'] for i in ids)
    assert starts[1]-starts[0]>=32
    for t in (48,80):assert any(s<=t<s+32 for s in starts)
    extra=[i for i,r in enumerate(augmented) if not r.get('descriptor_available',True)]
    assert set(extra)<=set(ids)
    assert all(reasons[i]=='observed_manipulation_anchor' for i in extra)
    # Placeholder descriptor values cannot determine selection of required rows.
    features[extra]=1e6
    changed,_=choose(augmented,features,2,7,2,{},base=0,random_fraction=0,
                     observed_anchors=anchors,preplanned_windows=plans)
    assert changed==ids
    for i in extra:
        assert augmented[i]['descriptor_row'] is None
        assert augmented[i]['candidate_origin']=='required_coverage_repair'


def test_new_configuration_does_not_reuse_original_panel_receipts(tmp_path):
    base=dict(root='/review',base_data='/base',curation=dict(output='/base'),
              panel_file='/base/panel.json',cpu_receipt_file='/base/cpu_checks.json')
    c=runtime_config(base,tmp_path)
    assert base['curation']['output']=='/base'
    assert c['curation_artifacts']=='/base'
    for key in ('panel_file','cpu_receipt_file','gpu_receipt_file','teacher_threshold_file'):
        assert c[key].startswith(str(tmp_path)+'/')
    assert c['readiness_dir']==str(tmp_path)


def test_revised_reader_uses_pinned_inventory_and_keeps_audit_sealed(tmp_path,monkeypatch):
    from vpp2.full_mask.data import WindowReader
    from vpp2.datasets.lerobot.utils import normalizer
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    runtime=tmp_path/'revision';runtime.mkdir()
    (artifacts/'inventory.json').write_text(json.dumps(dict(episodes=[
        dict(episode=1,role='fit'),dict(episode=2,role='monitor'),dict(episode=3,role='audit')])))
    stats={k:{'default':{'global_mean':[0.],'global_std':[1.]}} for k in ('action','state')}
    (tmp_path/'dataset_stats.json').write_text(json.dumps(stats))
    monkeypatch.setattr(normalizer,'SingleFieldLinearNormalizer',lambda *a,**k:object())
    c=dict(demonstrations=str(tmp_path),teacher_bundle=str(tmp_path),curation_artifacts=str(artifacts),
           curation=dict(schema='robodojo-curated-calibration-v3',output=str(runtime)))
    reader=WindowReader(c)
    assert reader.allowed_episodes=={1,2}
    with pytest.raises(ValueError,match='sealed'):
        reader.read(dict(episode=3),0)
