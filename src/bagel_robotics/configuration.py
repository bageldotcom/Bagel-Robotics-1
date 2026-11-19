"""Portable user configuration and the final method's numerical defaults."""
import json
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / "resources"


def read_json(path):
    return json.loads(Path(path).read_text())


def resource(name):
    path = RESOURCES / name
    if not path.is_file():
        raise ValueError("Missing packaged resource: " + name)
    return path


def load(path):
    path = Path(path).expanduser().resolve()
    config = read_json(path)
    if config.get("schema") != "bagel-reference-v1":
        raise ValueError("Expected schema bagel-reference-v1 in " + str(path))
    config["_config_dir"] = str(path.parent)
    config["_config_file"] = str(path)
    return config


def resolve(config, value, *, follow_symlinks=True):
    if value is None or value == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve() if follow_symlinks else Path(__import__('os').path.abspath(path))


def selected_recipe(config, stage):
    if stage in {"learn-masks", "prune"}:
        return "mask_learning"
    if stage == "recover":
        return "recovery"
    raise ValueError("Unsupported training stage: " + stage)


def recipe(name):
    if name not in {"mask_learning", "recovery"}:
        raise ValueError("Unknown recipe: " + name)
    return read_json(resource("recipes/" + name + ".json"))
