"""Lifecycle management for the five benign generator instances."""

from __future__ import annotations

import time
from pathlib import Path

from botocore.exceptions import ClientError

from aws_ids_testbed_10.config import load_config
from aws_ids_testbed_10.ec2_lab import (
    create_ec2_client,
    create_one_instance,
    refresh_instance_info,
    terminate_instance,
)
from aws_ids_testbed_10.inventory import load_inventory, save_inventory, update_instance


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


def _instance_state(ec2, instance_id: str) -> str | None:
    """Return an instance state, or None when AWS no longer knows the ID."""
    try:
        response = ec2.describe_instances(InstanceIds=[instance_id])
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"InvalidInstanceID.NotFound", "InvalidInstanceID.Malformed"}:
            return None
        raise

    reservations = response.get("Reservations", [])
    if not reservations:
        return None

    instances = reservations[0].get("Instances", [])
    if not instances:
        return None

    return instances[0].get("State", {}).get("Name")


def purge_benign_generators(project_root: Path) -> int:
    """Finish cleanup and remove stale entries using two bounded checks."""
    inventory = load_inventory(project_root)
    instances = inventory.get("instances", {})
    generator_entries = {
        role: instances.get(role)
        for role in BENIGN_GENERATOR_ROLES
        if instances.get(role)
    }

    if not generator_entries:
        print("[benign-generators] No stale benign-generator entries found.")
        return 0

    ec2 = create_ec2_client(project_root)
    pending_ids: dict[str, str] = {}

    for role, instance_data in generator_entries.items():
        instance_id = instance_data.get("instance_id")
        if not instance_id:
            print(f"[benign-generators] {role} has no instance ID; keeping entry.")
            pending_ids[role] = "missing-instance-id"
            continue

        state = _instance_state(ec2, instance_id)
        print(f"[benign-generators] {role}: {instance_id} state={state or 'absent'}")

        if state in {None, "terminated"}:
            continue

        if state == "shutting-down":
            # Already terminating; do not submit a duplicate request.
            pending_ids[role] = state
            continue

        try:
            response = ec2.terminate_instances(InstanceIds=[instance_id])
            current_state = response["TerminatingInstances"][0]["CurrentState"]["Name"]
            print(
                f"[benign-generators] {role}: termination requested "
                f"(state={current_state})."
            )
            pending_ids[role] = current_state
        except ClientError as exc:
            # A concurrent AWS transition may race the state check.
            print(f"[benign-generators] {role}: termination request deferred: {exc}")
            pending_ids[role] = "termination-request-error"

    max_checks = 2
    check_delay_seconds = 5

    for check_number in range(1, max_checks + 1):
        if check_number > 1:
            time.sleep(check_delay_seconds)

        remaining: dict[str, str] = {}
        for role, previous_state in pending_ids.items():
            instance_id = generator_entries[role].get("instance_id")
            if not instance_id:
                remaining[role] = previous_state
                continue

            state = _instance_state(ec2, instance_id)
            print(
                f"[benign-generators] Check {check_number}/{max_checks} "
                f"{role}: {instance_id} state={state or 'absent'}"
            )
            if state not in {None, "terminated"}:
                remaining[role] = state or previous_state

        pending_ids = remaining
        if not pending_ids:
            break

    if pending_ids:
        details = ", ".join(
            f"{role}={state}" for role, state in pending_ids.items()
        )
        print(
            "[benign-generators] Cleanup is still in progress after "
            f"{max_checks} checks ({check_delay_seconds}s apart): {details}."
        )
        print(
            "[benign-generators] Inventory entries were preserved. "
            "Run this command again after AWS reports terminated."
        )
        return 1

    for role in BENIGN_GENERATOR_ROLES:
        instances.pop(role, None)
    save_inventory(project_root, inventory)
    print("[benign-generators] Removed stale generator entries from inventory.")
    return 0
