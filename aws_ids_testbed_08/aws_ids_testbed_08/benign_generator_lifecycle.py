"""Lifecycle management for the five benign generator instances."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_08.config import load_config
from aws_ids_testbed_08.ec2_lab import (
    create_one_instance,
    refresh_instance_info,
    terminate_instance,
)
from aws_ids_testbed_08.inventory import load_inventory, update_instance


BENIGN_GENERATOR_ROLES = tuple(
    f"benign_generator_{index:02d}"
    for index in range(1, 6)
)


def _validate_generator_config(project_root: Path) -> None:
    """Verify that all five generator roles exist in config.yaml."""
    config = load_config(project_root)
    configured_roles = config.get("instances", {})

    missing_roles = [
        role
        for role in BENIGN_GENERATOR_ROLES
        if role not in configured_roles
    ]

    if missing_roles:
        raise ValueError(
            "Missing benign-generator roles in config.yaml: "
            + ", ".join(missing_roles)
        )


def create_benign_generators(project_root: Path) -> int:
    """Create all five generators and save them to inventory.yaml."""
    _validate_generator_config(project_root)

    inventory = load_inventory(project_root)
    existing_instances = inventory.get("instances", {})

    for role in BENIGN_GENERATOR_ROLES:
        if role in existing_instances:
            state = existing_instances[role].get("state", "unknown")
            raise RuntimeError(
                f"{role} already exists in inventory.yaml "
                f"with state={state}. "
                "Refresh or terminate it before creating another."
            )

    for role in BENIGN_GENERATOR_ROLES:
        print(f"[benign-generators] Creating {role}...")
        result = create_one_instance(project_root, role)
        update_instance(project_root, role, result)
        print(result)

    print("[benign-generators] Five generators created.")
    return 0


def refresh_benign_generators(project_root: Path) -> int:
    """Refresh state and IP addresses for all five generators."""
    inventory = load_inventory(project_root)
    instances = inventory.get("instances", {})

    for role in BENIGN_GENERATOR_ROLES:
        instance_data = instances.get(role)

        if not instance_data:
            raise RuntimeError(
                f"No inventory entry found for {role}. "
                "Create the generators first."
            )

        instance_id = instance_data["instance_id"]
        refreshed = refresh_instance_info(project_root, instance_id)
        updated = {**instance_data, **refreshed}

        update_instance(project_root, role, updated)
        print(role, updated)

    print("[benign-generators] Inventory refreshed.")
    return 0


def terminate_benign_generators(project_root: Path) -> int:
    """Terminate all five generators using their saved instance IDs."""
    inventory = load_inventory(project_root)
    instances = inventory.get("instances", {})

    for role in BENIGN_GENERATOR_ROLES:
        instance_data = instances.get(role)

        if not instance_data:
            raise RuntimeError(
                f"No inventory entry found for {role}."
            )

        instance_id = instance_data["instance_id"]
        terminated = terminate_instance(project_root, instance_id)
        updated = {**instance_data, **terminated}

        update_instance(project_root, role, updated)
        print(role, updated)

    print("[benign-generators] Termination requests submitted.")
    return 0
