"""Ensure an installed benchmark adapter uses the one canonical source package."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("install_bagel_adapter", ROOT / "scripts/install_adapter.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def xpolicylab(tmp_path):
    for name in ("XPolicyLab.py", "pyproject.toml", "client_server/ws/model_server.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return tmp_path


def test_adapter_installation_is_self_contained_and_idempotent(tmp_path):
    checkout = xpolicylab(tmp_path)
    installer.main(["--xpolicylab", str(checkout)])
    policy = checkout / "policy/Bagel_Robotics_1"
    package = policy / ".bagel_source"
    assert (package / "src/bagel_robotics/inference/policy.py").is_file()
    assert (package / "src/bagel_robotics/inference/configs/artifacts.json").is_file()
    assert (package / "requirements-inference.txt").is_file()
    assert (package / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert (policy / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    for notice in (ROOT / "licenses").iterdir():
        assert (package / "licenses" / notice.name).read_bytes() == notice.read_bytes()
        assert (policy / "licenses" / notice.name).read_bytes() == notice.read_bytes()
    assert not (policy / "runtime").exists()
    installer.main(["--xpolicylab", str(checkout)])
    (policy / "model.py").write_text("different adapter")
    with pytest.raises(SystemExit):
        installer.main(["--xpolicylab", str(checkout)])
    assert (policy / "model.py").read_text() == "different adapter"


def test_adapter_dry_run_writes_nothing(tmp_path):
    checkout = xpolicylab(tmp_path)
    installer.main(["--xpolicylab", str(checkout), "--dry-run"])
    assert not (checkout / "policy").exists()
