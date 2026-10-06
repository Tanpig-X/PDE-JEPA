"""Small YAML configuration loader shared by training entry points."""

import os
from pathlib import Path
import re

import yaml


def load_config(path, overrides=()):
    os.environ.setdefault("DATA_ROOT", str(Path("data").resolve()))
    os.environ.setdefault("OUTPUT_ROOT", str(Path("outputs").resolve()))
    config = yaml.safe_load(Path(path).read_text())
    for override in overrides:
        key, value = override.split("=", 1)
        parts, parent = key.split("."), config
        for part in parts[:-1]:
            parent = parent.setdefault(part, {})
        parent[parts[-1]] = yaml.safe_load(value)

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            value = os.path.expanduser(os.path.expandvars(value))
            if re.search(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", value):
                raise ValueError(f"Unset environment variable in {value!r}")
        return value

    return expand(config)
