import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release_tools import cli

class LauncherTests(unittest.TestCase):
    def test_rejects_relative_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            for path in ["relative","/a/../b"]:
                with self.assertRaises(ValueError):cli.contained(Path(d),path)
    def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/"escape").symlink_to("/tmp",target_is_directory=True)
            with self.assertRaises(ValueError):cli.contained(p,"/escape/test")
    def test_missing_assets_reported(self):
        self.assertIn("host_path",cli.check_asset("teacher",{"host_path":None})[0])
    def test_checkpoint_completeness(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/"weights.pt").write_bytes(b"123")
            cli.write(p/"COMPLETE.json",dict(update=55,identity="i",files={"weights.pt":{"size":3,"sha256":cli.file_hash(p/"weights.pt")}}))
            rec=dict(host_path=d,update=55,identity="i",required_files=["COMPLETE.json"])
            self.assertEqual(cli.check_asset("ckpt",rec,True),[])
            (p/"weights.pt").write_bytes(b"12")
            self.assertTrue(any("size" in x for x in cli.check_asset("ckpt",rec)))
    def test_wrong_checkpoint_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cli.write(Path(d)/"COMPLETE.json",dict(update=950,identity="other",files={}))
            errors=cli.check_asset("checkpoint",dict(host_path=d,update=1000,identity="expected"))
            self.assertEqual(len(errors),2)
    def test_weights_not_bundled(self):
        bad=[p for p in (cli.ROOT/"reproduction").rglob("*") if p.suffix in (".pt",".safetensors",".bin")]
        self.assertEqual(bad,[])
    def test_init_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"important").write_text("keep")
            with self.assertRaises(ValueError):cli.init_workspace(d)
            self.assertEqual((Path(d)/"important").read_text(),"keep")
    def test_stage_contracts(self):
        self.assertEqual(cli.STAGES["recovery-55-to-250"][1],"vpp2.flexible_mask.low_lr_entry")
        self.assertEqual(cli.STAGES["recovery-250-to-1000"][1],"vpp2.flexible_mask.recovery_extension_entry")
        self.assertIn("checkpoint55",cli.NEEDS["recovery-55-to-250"])
    def test_training_requires_allocation(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict("os.environ",{},clear=True),patch.object(cli,"doctor",return_value={"passed":True}):
                with self.assertRaisesRegex(ValueError,"Slurm"):
                    cli.run_stage(Path(d),"recovery-55-to-250","image","train","apptainer")
    def test_command_never_enables_upload(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);stage="recovery-55-to-250"
            config=cli.contained(p/"fs",cli.config_for(stage))
            cli.write(config,{"train":{"kv_weight":0.0},"recovery_lr_branch":{"load_scratch":"/tmp/vpp2"}})
            rec={"host_path":str(p),"container_path":"/input"}
            with patch.object(cli,"assets_for",return_value={"input":rec}):
                args=cli.command(p,stage,p/"image.sif")
            self.assertIn("WANDB_MODE=disabled",args)
            self.assertIn("--nproc_per_node=8",args)
            self.assertNotIn("shell",args)
            self.assertIn(str(p.resolve())+":/input:ro",args)
    def test_physical_export_is_one_process(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);stage="export-1500"
            cli.write(cli.contained(p/"fs",cli.config_for(stage)),{})
            rec={"host_path":str(p),"container_path":"/input"}
            with patch.object(cli,"assets_for",return_value={"input":rec}):
                args=cli.command(p,stage,p/"image.sif",action="export")
            self.assertNotIn("torch.distributed.run",args)
            self.assertTrue(args[-1].endswith("/control/export_selected.py"))
    def test_export_verification_uses_original_verifier(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);stage="export-1500"
            cli.write(cli.contained(p/"fs",cli.config_for(stage)),{})
            rec={"host_path":str(p),"container_path":"/input"}
            with patch.object(cli,"assets_for",return_value={"input":rec}):
                args=cli.command(p,stage,p/"image.sif",action="verify")
            self.assertIn("vpp2.flexible_mask.verify_export",args)
            self.assertNotIn("--nproc_per_node=8",args)
    def test_no_source_edits_on_command(self):
        config=cli.ROOT/"reproduction/runs.json"
        before=config.read_bytes()
        self.assertTrue(cli.config_for("mask-no-kv").endswith("/config.json"))
        self.assertEqual(config.read_bytes(),before)

if __name__=="__main__":unittest.main()
