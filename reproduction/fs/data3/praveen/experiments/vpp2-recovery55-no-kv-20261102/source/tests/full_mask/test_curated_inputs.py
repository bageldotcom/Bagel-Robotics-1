import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
import torch
from vpp2.compact_kv.model import prepare_batch
from vpp2.full_mask.curation import fingerprint_window,history_indices
from vpp2.full_mask.verify_data import require_training_readiness


class Scheduler:
    shift=1.;num_train_timesteps=1000
    def _phi(self,u,shift):return u
    def add_noise(self,x,n,t):return x*(1-t[:,None,None]/1000)+n*t[:,None,None]/1000
    def training_target(self,x,n,t):return n-x


class FakeReference:
    def __init__(self,inputs):
        self.inputs=inputs;self.calls=0
        self.vae=types.SimpleNamespace(temporal_downsample_factor=4)
        self.train_action_scheduler=Scheduler()
    def build_inputs(self,sample,**kwargs):self.calls+=1;return dict(self.inputs)
    def _build_mot_attention_mask(self,n,m,p,device):return torch.ones(n+m,n+m,device=device,dtype=torch.bool)


class Inputs(unittest.TestCase):
    def test_panel_identity_survives_json_integer_keys(self):
        from vpp2.full_mask.panel import digest
        value={'counts':{2:10,11:20,100:30}}
        self.assertEqual(digest(value),digest(json.loads(json.dumps(value))))

    def test_cached_preparation_matches_noise_and_does_not_mutate_cache(self):
        inputs=dict(input_latents=None,condition_latents=torch.zeros(1,16,3,30,52),
                    action=torch.zeros(1,32,16),num_video_frames=17)
        ref=FakeReference(inputs);sample={'pilot_position':torch.tensor([12345])}
        live=prepare_batch(ref,sample,100008)
        reuse=prepare_batch(ref,sample,100008,prepared_inputs=inputs)
        self.assertEqual(ref.calls,1);self.assertIn('input_latents',inputs)
        for a,b in zip(live[1:],reuse[1:]):self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):prepare_batch(ref,sample,1,prepared_inputs=inputs,encode_video_target=True)
    def test_different_memory_history_is_not_an_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.zeros((64,16),np.float32);a[:,3]=1;a[:,11]=1
            f=np.zeros((64,3,240,416),np.uint8)
            for ep in ('a','b'):
                p=Path(d)/ep;p.mkdir();np.save(p/'actions.npy',a);np.save(p/'states.npy',a)
                images=f.copy();images[0,0 if ep=='a' else 2]=255;np.save(p/'frames.npy',images)
            r=dict(prompt='remember the colour',frame=32,history_indices=history_indices(32))
            one=fingerprint_window({'demonstrations':d},dict(r,episode_path='a'))
            two=fingerprint_window({'demonstrations':d},dict(r,episode_path='b'))
            self.assertNotEqual(one,two)
    def test_readiness_never_accepts_a_review_page_as_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            c={'curation':{'output':d}};panel={'identity':'test'}
            for name in ('cpu_checks.json','visual_review_receipt.json','model_probe_receipt.json'):
                Path(d,name).write_text(json.dumps({'panel_identity':'test','passed':True}))
            require_training_readiness(c,panel)
            Path(d,'visual_review_receipt.json').write_text(json.dumps({'panel_identity':'test','status':'pending'}))
            with self.assertRaises(RuntimeError):require_training_readiness(c,panel)
            Path(d,'visual_review_receipt.json').write_text(json.dumps({'panel_identity':'wrong','passed':True}))
            with self.assertRaises(RuntimeError):require_training_readiness(c,panel)


if __name__=='__main__':unittest.main()
