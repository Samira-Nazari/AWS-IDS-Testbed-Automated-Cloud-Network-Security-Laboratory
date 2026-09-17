"""Save and load EC2 inventory information."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def inventory_path(project_root: Path) -> Path:
    """Return the inventory.yaml path."""
    return project_root / "inventory.yaml"


def save_inventory(project_root: Path, inventory: dict[str, Any]) -> None:
    """Save EC2 instance information to inventory.yaml."""
    path = inventory_path(project_root)

    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(inventory, file_obj, sort_keys=False)


def load_inventory(project_root: Path) -> dict[str, Any]:
    """Load inventory.yaml if it exists."""
    path = inventory_path(project_root)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def update_instance(project_root: Path, role: str, instance_data: dict[str, Any]) -> None:
    """Save one created instance into inventory.yaml."""
    inventory = load_inventory(project_root)

    # Keep all EC2 instances under one top-level key.
    if "instances" not in inventory or not isinstance(inventory["instances"], dict):
        inventory["instances"] = {}

    # Example roles are victim, attacker, and ids.
    inventory["instances"][role] = instance_data

    save_inventory(project_root, inventory)
