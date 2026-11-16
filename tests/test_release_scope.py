import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from release_tools import cli
spec=importlib.util.spec_from_file_location('prepare_data_release',ROOT/'scripts/prepare_data.py')
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)


class ReleaseScopeTests(unittest.TestCase):
    def test_all_exposed_training_stages_disable_kv(self):
        for stage in cli.STAGES:
            c=cli.read(cli.contained(ROOT/'reproduction/fs',cli.config_for(stage)))
            cli.require_no_kv_training(stage,c)
            if stage!='export-1500':self.assertEqual(c['train']['kv_weight'],0)
        self.assertNotIn('recovery-initial',cli.STAGES)
        self.assertNotIn('mask-from-start',cli.STAGES)

    def test_guard_rejects_nonzero_or_missing_loss(self):
        for value in [0.05,1,-1,None,False,'0']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                cli.require_no_kv_training('mask-no-kv',{'train':{'kv_weight':value}})

    def test_episode_hash_check_detects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'episodes/0';p.mkdir(parents=True)
            (p/'actions.npy').write_bytes(b'original')
            row={'id':'0','path':'episodes/0','files':{'actions.npy':data.sha256(p/'actions.npy')}}
            self.assertEqual(data.verify_episode(d,row),[])
            (p/'actions.npy').write_bytes(b'changed')
            self.assertTrue(data.verify_episode(d,row))

    def test_data_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            for value in ['/etc/passwd','../outside']:
                with self.assertRaises(ValueError):data.safe_child(d,value)

    def test_conditioning_and_normalization_are_original(self):
        manifest=json.loads((ROOT/'data/native-manifest.json').read_text())
        for row in {r['text']:r for r in manifest['episodes']}.values():
            self.assertEqual(data.sha256(ROOT/'data'/row['text']),row['text_sha256'])
        self.assertEqual(data.sha256(ROOT/'data/dataset_stats.json'),
            '2649d4e3ff3d6d400c9953b7c20c79541474f9c38a8a9916b9caacecd8578f16')

if __name__=='__main__':unittest.main()
