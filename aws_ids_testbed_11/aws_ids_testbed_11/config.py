"""Load project configuration from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(project_root: Path) -> dict[str, Any]:
    """Read config.yaml and return its values."""
    config_path = project_root / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}
