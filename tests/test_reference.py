"""CPU contracts for the researcher interface; no numerical runtime required."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import bagel_robotics
from bagel_robotics import configuration, workspace


PACKAGE_PARENT = Path(bagel_robotics.__file__).resolve().parent.parent
PINNED_DATA = {
    "selected_panel.json.gz": "933aa8a10bc826417dc05d97e462efdd339b8292ff9d1978dec60261e974e538",
    "native_manifest.json.gz": "72edc43dd50d6c830fae2cbd0322842df05a7e6dcf08b79b316e5eb3769fc945",
    "normalization.json": "2649d4e3ff3d6d400c9953b7c20c79541474f9c38a8a9916b9caacecd8578f16",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def directory_state(root):
    """Detect writes, removals and symlink changes in a temporary fixture."""
    return {
        str(path.relative_to(root)): ("link", str(path.readlink())) if path.is_symlink()
        else ("file", digest(path.read_bytes())) if path.is_file() else ("directory",)
        for path in root.rglob("*")
    }


class ResearchInterfaceTests(unittest.TestCase):
    def run_cli(self, directory, *arguments):
        # A fresh process catches accidental eager imports even on hosts where
        # numpy or torch happens to be installed. Use the imported distribution
        # location, so the same tests also exercise an installed wheel.
        script = """
import importlib.abc
import runpy
import subprocess
import sys
class BlockNumericalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'transformers', 'deepspeed'}:
            raise RuntimeError('Unexpected eager numerical import: ' + fullname)
sys.meta_path.insert(0, BlockNumericalRuntime())
def reject_child_process(*args, **kwargs):
    raise RuntimeError('Read-only CLI inspection started an external process')
subprocess.run = reject_child_process
sys.argv = ['bagel_robotics'] + sys.argv[1:]
runpy.run_module('bagel_robotics', run_name='__main__')
"""
        env = dict(os.environ, PYTHONPATH=str(PACKAGE_PARENT), PYTHONDONTWRITEBYTECODE="1",
                   CUDA_VISIBLE_DEVICES="0")
        return subprocess.run(
            [sys.executable, "-c", script, *arguments], cwd=directory,
            env=env, text=True, capture_output=True, timeout=30,
        )

    def test_cli_help_is_available_without_a_numerical_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, "--help")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for command in ("prepare-data", "select-data", "learn-masks", "prune", "recover", "evaluate", "doctor"):
            self.assertIn(command, result.stdout)

    def test_relative_paths_are_anchored_to_config_not_process_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "settings" / "research.json"
            write_json(path, {"schema": "bagel-reference-v1", "workspace": "../runs/new"})
            config = configuration.load(path)
            self.assertEqual(workspace.workspace_path(config), root / "runs" / "new")
            self.assertIsNone(configuration.resolve(config, None))
            self.assertEqual(configuration.resolve(config, str(root / "assets")), root / "assets")
            write_json(path, {"schema": "another-project"})
            with self.assertRaisesRegex(ValueError, "schema"):
                configuration.load(path)

    def test_only_supported_stages_select_a_recipe(self):
        self.assertEqual(configuration.selected_recipe({}, "recover"), "recovery")
        self.assertEqual(configuration.selected_recipe({}, "prune"), "mask_learning")
        with self.assertRaisesRegex(ValueError, "Unsupported training stage"):
            configuration.selected_recipe({}, "replace-model")
        with self.assertRaisesRegex(ValueError, "Unknown recipe"):
            configuration.recipe("../assets")

    def test_logical_paths_reject_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "workspace"
            outside = Path(directory).resolve() / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "escape").symlink_to(outside, target_is_directory=True)
            self.assertEqual(workspace.contained(root, "/new/output"), root / "new/output")
            for logical in ("relative/output", "/../outside", "/nested/../../outside", "/escape/artifact"):
                with self.subTest(logical=logical), self.assertRaises(ValueError):
                    workspace.contained(root, logical)

    def test_training_dry_runs_create_no_workspace_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config_path = root / "research.json"
            write_json(config_path, {
                "schema": "bagel-reference-v1",
                "workspace": "new-workspace",
                "paths": {name: "assets/" + name for name in (
                    "demonstrations", "encoders", "base_model", "compact_model"
                )},
                "execution": {"engine": "apptainer", "image": "images/training.sif"},
            })
            before = directory_state(root)
            for command, action in (("learn-masks", "train"), ("prune", "export"), ("recover", "train")):
                with self.subTest(command=command):
                    result = self.run_cli(root, command, "--config", str(config_path), "--action", action, "--dry-run")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("bagel_robotics.", result.stdout)
                    self.assertEqual(directory_state(root), before)

    def test_installed_training_launch_mounts_only_the_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            site_packages = root / "venv/lib/python/site-packages"
            package = site_packages / "bagel_robotics"
            config = {
                "_config_dir": str(root), "workspace": "workspace",
                "paths": {name: "inputs/" + name for name in ("demonstrations", "encoders", "base_model")},
                "execution": {"engine": "apptainer", "image": "training.sif"},
            }
            with mock.patch.object(workspace, "__file__", str(package / "workspace.py")):
                command = workspace.command(config, "learn-masks")
            mounts = [command[index + 1] for index, value in enumerate(command) if value == "--bind"]
            sources = [entry.split(":", 1)[0] for entry in mounts]
            self.assertIn(str(package), sources)
            self.assertNotIn(str(site_packages), sources)
            self.assertTrue(any(entry.startswith(str(package) + ":") and entry.endswith(":ro") for entry in mounts))

    def test_default_native_dry_run_requires_no_container_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "research.json"
            write_json(path, {
                "schema": "bagel-reference-v1", "workspace": "workspace",
                "paths": {name: "inputs/" + name for name in ("demonstrations", "encoders", "base_model")},
            })
            before = directory_state(root)
            result = self.run_cli(root, "learn-masks", "--config", str(path), "--dry-run")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertIn("argv", report)
            self.assertIn("torch.distributed.run", report["argv"])
            self.assertNotIn("--bind", report["argv"])
            self.assertNotIn("apptainer", report["argv"])
            self.assertEqual(directory_state(root), before)

    def test_evaluation_retains_the_configured_venv_executable_path(self):
        from bagel_robotics.cli import evaluation_commands
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable = root / "policy-env/bin/python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            config = {"_config_dir": str(root), "paths": {"policy": "policy", "python": "policy-env/bin/python"}}
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                download = evaluation_commands(config, "download")[0]
                smoke = evaluation_commands(config, "smoke")
            self.assertEqual(download["argv"][0], str(executable))
            for entry in smoke:
                self.assertEqual(entry["argv"][-2:], [str(executable.parent.parent)] * 2)

    def test_missing_assets_and_changed_training_state_are_actionable(self):
        self.assertEqual(workspace.check_asset("base_model", {"host_path": None}),
                         ["Set paths.base_model in your configuration"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            missing = workspace.check_asset("base_model", {"host_path": root / "missing"})
            self.assertTrue(any("base_model: missing" in error for error in missing))
            (root / "model.bin").write_bytes(b"small fixture")
            record = {"host_path": root, "required_files": ["model.bin"], "identity": "expected-parent"}
            write_json(root / "COMPLETE.json", {"identity": "different-parent", "files": {"model.bin": {"size": 13}}})
            self.assertIn("base_model: wrong identity", workspace.check_asset("base_model", record))
            write_json(root / "COMPLETE.json", {"identity": "expected-parent", "files": {"model.bin": {"size": 99}}})
            self.assertIn("base_model: wrong size model.bin", workspace.check_asset("base_model", record))

    def test_data_and_evaluation_dry_runs_need_no_assets_or_external_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config_path = root / "research.json"
            write_json(config_path, {
                "schema": "bagel-reference-v1", "workspace": "workspace",
                "data": {"output": "converted", "selection_output": "selected"},
                "paths": {"policy": "adapter", "python": "environment/bin/python"},
            })
            before = directory_state(root)
            cases = (("prepare-data", "--episodes", "0"), ("select-data",),
                     ("evaluate", "--operation", "download"), ("evaluate", "--operation", "smoke"))
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    result = self.run_cli(root, *arguments, "--config", str(config_path), "--dry-run")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(directory_state(root), before)

    def test_full_asset_verification_checks_weights_and_rejects_escaped_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "weight.bin").write_bytes(b"changed")
            record = {"host_path": root, "weight_sha256": {"weight.bin": digest(b"original")}}
            self.assertEqual(workspace.check_asset("base_model", record), [])
            self.assertIn("base_model: checksum mismatch weight.bin",
                          workspace.check_asset("base_model", record, full_hash=True))
            write_json(root / "COMPLETE.json", {"files": {"../outside": {"size": 1}}})
            with self.assertRaises(ValueError):
                workspace.check_asset("base_model", {"host_path": root})

    def test_new_workspace_starts_without_model_state_or_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = {"_config_dir": str(root), "workspace": "run"}
            result = workspace.init(config)
            self.assertFalse(result["model_weights_included"])
            self.assertFalse(json.loads((root / "run/workspace.json").read_text())["qualified"])
            self.assertEqual(list((root / "run").rglob("verified-*.json")), [])
            self.assertEqual(list((root / "run").rglob("COMPLETE.json")), [])
            self.assertFalse((root / "run/fs").exists())
            self.assertEqual(digest((root / "run/inputs/panel.json").read_bytes()), PINNED_DATA["selected_panel.json.gz"])
            self.assertEqual(digest((root / "run/inputs/normalization.json").read_bytes()), PINNED_DATA["normalization.json"])
            before = directory_state(root / "run")
            with self.assertRaisesRegex(ValueError, "new empty workspace"):
                workspace.init(config)
            self.assertEqual(directory_state(root / "run"), before)

    def qualification_fixture(self, root):
        package = root / "installed/bagel_robotics"
        package.mkdir(parents=True)
        (package / "train.py").write_text("value = 1\n")
        image = root / "training.sif"
        image.write_bytes(b"container fixture")
        paths = {}
        for name in ("demonstrations", "base_model", "encoders", "compact_model"):
            path = root / "inputs" / name
            path.mkdir(parents=True)
            (path / "input.bin").write_bytes(b"input fixture")
            paths[name] = str(path)
        config = {
            "_config_dir": str(root), "workspace": "workspace", "paths": paths,
            "execution": {"image": str(image)},
        }
        # Fingerprinting treats these as opaque bytes; no alternate dataset is
        # passed to a scientific reader in these qualification tests.
        inputs = root / "workspace/inputs"
        inputs.mkdir(parents=True)
        (inputs / "panel.json").write_bytes(b"selected-window fixture")
        (inputs / "normalization.json").write_bytes(b"normalization fixture")
        write_json(root / "workspace/mask_learning/config.json", workspace.resolved_recipe(config, "learn-masks"))
        return config, package

    def test_training_and_pruning_refuse_an_unqualified_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            config, package = self.qualification_fixture(Path(directory).resolve())
            with mock.patch.object(workspace, "doctor", return_value={"passed": True}), \
                    mock.patch.object(workspace.subprocess, "run") as launch:
                for stage, action in (("learn-masks", "train"), ("prune", "select"), ("prune", "export"), ("recover", "train")):
                    with self.subTest(stage=stage, action=action), self.assertRaisesRegex(ValueError, "verify action"):
                        workspace.run(config, stage, action)
                launch.assert_not_called()

    def test_qualification_expires_when_code_recipe_image_input_or_workspace_changes(self):
        for changed in ("source", "recipe", "image", "input", "staged_panel", "staged_recipe", "workspace"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                config, package = self.qualification_fixture(root)
                with mock.patch.object(workspace, "__file__", str(package / "workspace.py")):
                    receipt = workspace.workspace_path(config) / "logs/verified-mask_learning.json"
                    approved = {"passed": True, "fingerprint": workspace.qualification_fingerprint(config, "learn-masks")}
                    write_json(receipt, approved)
                    workspace.require_qualification(config, "learn-masks")
                    if changed == "source":
                        (package / "train.py").write_text("value = 2\n")
                    elif changed == "recipe":
                        config["recipe_overrides"] = {"mask_learning": {"train": {"video_lr": 2e-6}}}
                    elif changed == "image":
                        (root / "training.sif").write_bytes(b"changed container")
                    elif changed == "input":
                        path = root / "inputs/base_model/input.bin"
                        prior = path.stat()
                        path.write_bytes(b"changed input")
                        os.utime(path, ns=(prior.st_atime_ns, prior.st_mtime_ns + 1_000_000_000))
                    elif changed == "staged_panel":
                        (root / "workspace/inputs/panel.json").write_bytes(b"changed selection fixture")
                    elif changed == "staged_recipe":
                        path = root / "workspace/mask_learning/config.json"
                        value = json.loads(path.read_text())
                        value["seed"] += 1
                        write_json(path, value)
                    else:
                        config["workspace"] = "different-workspace"
                        write_json(workspace.workspace_path(config) / "logs/verified-mask_learning.json", approved)
                    with self.assertRaisesRegex(ValueError, "stale"):
                        workspace.require_qualification(config, "learn-masks")


class ReleasedDataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {}
        for name in PINNED_DATA:
            raw = configuration.resource(name).read_bytes()
            cls.data[name] = gzip.decompress(raw) if name.endswith(".gz") else raw
        cls.panel = json.loads(cls.data["selected_panel.json.gz"])
        cls.manifest = json.loads(cls.data["native_manifest.json.gz"])

    def test_panel_manifest_and_normalization_retain_released_bytes(self):
        for name, expected in PINNED_DATA.items():
            with self.subTest(resource=name):
                self.assertEqual(digest(self.data[name]), expected)
        pins = configuration.read_json(configuration.resource("data.json"))
        self.assertEqual(pins["panel_sha256"], PINNED_DATA["selected_panel.json.gz"])
        self.assertEqual(pins["normalization_sha256"], PINNED_DATA["normalization.json"])

    def test_fitting_monitoring_and_official_episodes_remain_separate(self):
        panel = self.panel
        partitions = defaultdict(set)
        for row in panel["episode_partitions"]:
            partitions[row["role"]].add(row["episode"])
        self.assertEqual({name: len(rows) for name, rows in partitions.items()},
                         {"fit": 2906, "monitor": 280, "audit": 280})
        self.assertEqual(len(set.union(*partitions.values())), sum(map(len, partitions.values())))
        official = set(panel["official_validation_episodes"])
        self.assertFalse(official & set.union(*partitions.values()))
        self.assertEqual(len(official), 34)
        native = {int(row["id"]): row for row in self.manifest["episodes"]}
        self.assertEqual(len(native), 3500)
        self.assertEqual({episode for episode, row in native.items() if row["split"] == "validation"}, official)

    def test_selected_windows_keep_complete_actions_and_task_coverage(self):
        rows = self.panel["windows"]
        self.assertEqual(Counter(row["role"] for row in rows), {"selection_initial": 30000, "monitor": 1119})
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertEqual(len({(row["episode"], row["frame"]) for row in rows}), len(rows))
        native = {int(row["id"]): row for row in self.manifest["episodes"]}
        by_episode = defaultdict(list)
        for row in rows:
            episode = native[row["episode"]]
            self.assertEqual(episode["split"], "train")
            self.assertEqual(row["task"], episode["task"])
            self.assertGreaterEqual(row["frame"], 0)
            self.assertLessEqual(row["frame"] + 32, episode["length"])
            by_episode[row["episode"]].append(row["frame"])
        for starts in by_episode.values():
            ordered = sorted(starts)
            self.assertTrue(all(right - left >= 32 for left, right in zip(ordered, ordered[1:])))
        for role in ("selection_initial", "monitor"):
            self.assertEqual({row["task"] for row in rows if row["role"] == role}, set(self.panel["tasks"]))


if __name__ == "__main__":
    unittest.main()
