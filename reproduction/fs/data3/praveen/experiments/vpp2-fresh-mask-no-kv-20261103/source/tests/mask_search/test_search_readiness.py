import copy
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import torch
from vpp2.full_mask.calibration import sampling_plan,validate_rows
from vpp2.learned_width_kv.reliability import thresholds_from_rows,weights,kv_losses
from vpp2.full_mask.search_quality import quality_summary,better,schedule_end,stable_history,sensitivity_check
from vpp2.full_mask.monitor import overlap,confirm_full
from vpp2.full_mask.native_monitor import select_windows
from vpp2.full_mask.conditioning_cache import ConditioningCache


class CalibrationTests(unittest.TestCase):
    def panel(self):
        rows=[]
        for task,n in [('memory',83),('precision',84)]:
            for ep in range(n):
                for frame in [0,64]:rows.append(dict(id=len(rows),task=task,episode=ep+(0 if task=='memory' else 1000),frame=frame,role='selection_initial'))
        rows.append(dict(id=999999,task='memory',episode=999999,frame=32,role='monitor'))
        return dict(tasks=['memory','precision'],windows=rows)
    def test_actual_episode_counts_all_four_strata(self):
        panel=self.panel();selected,counts=sampling_plan(panel)
        self.assertEqual(counts[('memory',0)],166);self.assertEqual(counts[('precision',3)],168)
        rows=[dict(window_id=r['id'],task=r['task'],episode=r['episode'],stratum=s,teacher_error=float(i%17)) for i,r in enumerate(selected) for s in range(4)]
        table=validate_rows(rows,selected,counts,panel)
        self.assertEqual(table['memory'][2]['count'],166)
        for corrupt in (rows[:-1],rows+[rows[0]],rows+[dict(rows[0],window_id=999999)]):
            with self.assertRaises(ValueError):validate_rows(corrupt,selected,counts,panel)
        bad=copy.deepcopy(rows);bad[0]['teacher_error']=float('nan')
        with self.assertRaises(ValueError):validate_rows(bad,selected,counts,panel)
    def test_legacy_exact_64_contract_unchanged(self):
        rows=[dict(task='a',stratum=s,teacher_error=1.) for s in range(4) for i in range(64)]
        self.assertEqual(thresholds_from_rows(rows)['a'][0]['q75'],1.)
        with self.assertRaises(ValueError):thresholds_from_rows(rows+[rows[0]])
    def test_rejected_teacher_has_no_kv_gradient(self):
        thresholds={'a':[dict(median=1.,q75=2.,q95=3.)]*4}
        w,_=weights(torch.tensor([2.]),torch.tensor([1.]),torch.tensor([3.]),['a'],[0],thresholds)
        terms=torch.ones(1,40,2,requires_grad=True)
        _,loss,_=kv_losses(terms,w);loss.backward()
        self.assertEqual(w.item(),0.);self.assertEqual(terms.grad.count_nonzero(),0)


class QualityTests(unittest.TestCase):
    def rows(self,ratio=1.):
        return [dict(task=task,suite=suite,episode=ep,frame=frame,teacher_error=scale*(1+ep/10),hard_error=ratio*scale*(1+ep/10))
            for task,suite,scale in [('small','memory',1.),('large','precision',100.)] for ep in range(8) for frame in [0,32]]
    def test_task_normalization_and_episode_not_window_confidence(self):
        r=quality_summary(self.rows(1.1));self.assertAlmostEqual(r['mean_task_ratio'],1.1)
        self.assertFalse(r['no_resolved_regressions']);self.assertEqual(r['by_task']['small']['episodes'],8)
        self.assertTrue(quality_summary(self.rows(1.))['noninferiority_supported'])
        self.assertTrue(better(quality_summary(self.rows(.9)),r))
        with self.assertRaises(ValueError):quality_summary(self.rows()+self.rows()[:1])
    def test_stopping_cannot_use_pre_ramp_comparisons(self):
        c=dict(mask=dict(entry_updates=35,budget_ramp_updates=280),train=dict(kv_ramp_updates=100,minimum_stop_update=350,stable_checks=5))
        self.assertEqual(schedule_end(c),560)
        history=[dict(update=x,previous_update=x-70,stable=True) for x in range(140,561,70)]
        self.assertFalse(stable_history(history,560,c))
        history.extend(dict(update=x,previous_update=x-70,stable=True) for x in range(630,911,70))
        self.assertTrue(stable_history(history,910,c))
    def test_numerical_dust_not_perfect_convergence(self):
        c={'train':dict(suite_sensitivity_top_overlap_min=.9,noise_aware_sensitivity=True,sensitivity_half_overlap_min=.5)}
        a=dict(sensitivity_groups=[dict(name='hidden',active=True,top=list(range(10)),half_overlap=.7),dict(name='ffn_0',active=False,top=[1],half_overlap=0.)])
        b=copy.deepcopy(a);b['sensitivity_groups'][1]['top']=[99]
        self.assertTrue(sensitivity_check(a,b,c,overlap)['passed'])
        b['sensitivity_groups'][0]['half_overlap']=.1
        self.assertFalse(sensitivity_check(a,b,c,overlap)['passed'])
        a['sensitivity_groups'][0]['active']=False;b=copy.deepcopy(a)
        self.assertFalse(sensitivity_check(a,b,c,overlap)['passed'])
    def test_noise_adjustment_does_not_accept_changed_strong_channels(self):
        c={'train':dict(suite_sensitivity_top_overlap_min=.9,noise_aware_sensitivity=True,sensitivity_half_overlap_min=.5)}
        a=dict(sensitivity_groups=[dict(name='hidden',active=True,top=[0,1],half_overlap=1.)]);b=copy.deepcopy(a)
        b['sensitivity_groups'][0]['top']=[8,9]
        self.assertFalse(sensitivity_check(a,b,c,overlap)['passed'])
    def test_full_panel_stability_cannot_hide_clear_action_regression(self):
        c={'train':dict(hidden_churn_max=.01,ffn_churn_max=.03,suite_error_relative_change_max=.02,suite_sensitivity_top_overlap_min=.9,early_stop_quality_check=True)}
        report=dict(full=True,update=1000,selection=dict(hidden_indices=[0,1],ffn_indices=[[0,1],[0,1]]),
            by_suite={'memory':dict(hard_error=2.,hidden_top=[0],ffn_top=[[0]],half_hidden_overlap=1.,half_ffn_overlap=[1.])},
            quality=quality_summary(self.rows(2.)))
        self.assertFalse(confirm_full(report,copy.deepcopy(report),c)['passed'])


class CacheTests(unittest.TestCase):
    def test_atomic_roundtrip_corruption_and_config_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            c=ConditioningCache(dict(runtime_conditioning_cache=d,seed=1))
            tensors=dict(action=torch.randn(1,32,16,dtype=torch.bfloat16),condition_latents=torch.randn(1,16,3,30,52),input_latents=None,num_video_frames=17)
            self.assertIsNone(c.load(4));c.save(4,tensors);loaded=c.load(4)
            self.assertTrue(torch.equal(loaded['action'],tensors['action']))
            # Repeated publication does not replace another completed entry.
            c.save(4,dict(tensors,action=torch.zeros_like(tensors['action'])))
            self.assertTrue(torch.equal(c.load(4)['action'],tensors['action']))
            other=ConditioningCache(dict(runtime_conditioning_cache=d,seed=2));self.assertIsNone(other.load(4))
            path=c.path(4);raw=bytearray(path.read_bytes());raw[-10]^=1;path.write_bytes(raw)
            with self.assertRaises(ValueError):c.load(4)


class NativePanelTests(unittest.TestCase):
    def test_independent_episodes_and_verified_window_protected(self):
        rows=[dict(task=t,episode=e+(0 if t=='a' else 100),id=len(t)+e*2+f,frame=f,role='monitor',critical_steps=3 if e==7 and f==32 else None)
              for t in ['a','b'] for e in range(8) for f in [0,32]]
        panel=dict(tasks=['a','b'],windows=rows);c=dict(seed=100008,train=dict(native_windows_per_task=4))
        a=select_windows(panel,c);self.assertEqual(a,select_windows(panel,c))
        self.assertEqual(len(a),8);self.assertEqual(len({r['episode'] for r in a}),8)
        self.assertEqual(sum((r.get('critical_steps') or 0)>0 for r in a),2)


class ExportTests(unittest.TestCase):
    def test_final_export_uses_best_checkpoint_and_retries_idempotently(self):
        from vpp2.full_mask.controller import Controller
        from vpp2.full_mask.train import save,finalize_selection
        from vpp2.full_mask.panel import file_hash
        c=dict(seed=100008,mask=dict(keep_hidden=3,keep_ffn_total=39*2,minimum_ffn_per_layer=2))
        controller=Controller(c)
        with torch.no_grad():
            controller.hidden_scores.fill_(-2);controller.hidden_scores[7:10]=2
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'runs/masks_000560.pt'
            save(path,dict(identity='experiment',update=560,selection_config=c,controller=controller.state_dict()))
            best=dict(update=560,checkpoint=str(path),checkpoint_sha256=file_hash(path),quality=dict(mean_task_ratio=.9,worst_task_ratio=1.1))
            finalize_selection(root,'experiment',best,None,1400,False,1400)
            finalize_selection(root,'experiment',best,None,1400,False,1400)
            result=json.loads((root/'runs/final_selection.json').read_text())
            self.assertEqual(result['selected_update'],560);self.assertEqual(result['hidden_indices'],[7,8,9])
            self.assertFalse(result['policy_retention_established'])
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):finalize_selection(root,'experiment',best,None,1400,False,1400)

class PreparedCacheTests(unittest.TestCase):
    def test_cached_encoders_preserve_noise_targets_and_refresh_each_position(self):
        from types import SimpleNamespace
        from vpp2.full_mask.data import prepared
        class Scheduler:
            shift=1.;num_train_timesteps=1000
            def _phi(self,u,shift):return u
            def add_noise(self,a,n,t):return a*(1-t[:,None,None]/1000)+n*t[:,None,None]/1000
            def training_target(self,a,n,t):return n-a
        with tempfile.TemporaryDirectory() as d:
            cache=ConditioningCache(dict(runtime_conditioning_cache=d,seed=100008))
            inputs=dict(input_latents=None,condition_latents=torch.randn(1,16,3,30,52),action=torch.randn(1,32,16),num_video_frames=17)
            calls=[]
            def build(sample,**kwargs):
                calls.append(1);return {k:v.clone() if torch.is_tensor(v) else v for k,v in inputs.items()}
            reference=SimpleNamespace(device=torch.device('cpu'),build_inputs=build,vae=SimpleNamespace(temporal_downsample_factor=4),
                train_action_scheduler=Scheduler(),_full_mask_conditioning_cache=cache,
                _build_mot_attention_mask=lambda v,a,p,device:torch.ones(v+a,v+a,device=device,dtype=torch.bool))
            live=prepared(reference,dict(pilot_position=torch.tensor([123]),_conditioning_key=torch.tensor([7])),100008,[2])
            saved=cache.load(7)
            cached=prepared(reference,dict(_native_prepared=saved,pilot_position=torch.tensor([123])),100008,[2])
            for a,b in zip(live,cached):
                if isinstance(a,dict):
                    for key in a:
                        self.assertTrue(torch.equal(a[key],b[key]) if torch.is_tensor(a[key]) else a[key]==b[key])
                else:self.assertTrue(torch.equal(a,b))
            other=prepared(reference,dict(_native_prepared=cache.load(7),pilot_position=torch.tensor([124])),100008,[2])
            self.assertFalse(torch.equal(cached[1],other[1]));self.assertFalse(torch.equal(cached[3],other[3]))
            self.assertEqual(len(calls),1)

class SchedulerTests(unittest.TestCase):
    def test_old_slurm_stop_request_does_not_stop_resumed_job(self):
        from unittest.mock import patch
        from vpp2.full_mask.train import scheduler_stop_requested
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{'SLURM_JOB_ID':'222'}):
            root=Path(d);(root/'control').mkdir();p=root/'control/stop_request.json'
            p.write_text(json.dumps(dict(job_id='111')));self.assertFalse(scheduler_stop_requested(root))
            p.write_text(json.dumps(dict(job_id='222')));self.assertTrue(scheduler_stop_requested(root))

if __name__=='__main__':unittest.main()
