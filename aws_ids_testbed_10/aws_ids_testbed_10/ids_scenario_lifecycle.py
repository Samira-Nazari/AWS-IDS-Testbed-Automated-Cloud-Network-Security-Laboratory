"""Manage lifecycle markers for one IDS scenario execution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from aws_ids_testbed_10.ids_inference_workspace import (
    validate_runtime_name,
)


@dataclass(frozen=True)
class ScenarioLifecyclePaths:
    """State-marker paths for one scenario execution."""

    state_directory: Path
    input_finished_path: Path
    complete_path: Path


@dataclass(frozen=True)
class ScenarioDrainState:
    """Processing state after scenario input has finished."""

    execution_id: str
    input_finished: bool

    received_transactions: tuple[str, ...]
    completed_transactions: tuple[str, ...]

    pending_conversions: tuple[str, ...]
    pending_inference: tuple[str, ...]

    conversion_failures: tuple[str, ...]
    inference_failures: tuple[str, ...]

    pipeline_busy: bool
    latest_activity_ns: int

    @property
    def ready_for_quiet_period(self) -> bool:
        """Return whether the five-minute timer may start."""

        return (
            self.input_finished
            and not self.pending_conversions
            and not self.pending_inference
            and not self.pipeline_busy
        )


def _utc_timestamp() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def build_scenario_lifecycle_paths(
    *,
    base_directory: Path,
    execution_id: str,
) -> ScenarioLifecyclePaths:
    """Build lifecycle paths without creating marker files."""

    execution_id = validate_runtime_name(
        execution_id,
        field_name="execution_id",
    )

    state_directory = (
        Path(base_directory)
        / "state"
        / "scenario_runs"
    )

    return ScenarioLifecyclePaths(
        state_directory=state_directory,
        input_finished_path=(
            state_directory / f"{execution_id}.end.json"
        ),
        complete_path=(
            state_directory / f"{execution_id}.complete.json"
        ),
    )


def current_scenario_input_is_finished(
    *,
    base_directory: Path,
) -> bool:
    """Return whether Slurm ended input for the current run."""

    base_directory = Path(base_directory)

    execution_id_path = (
        base_directory
        / "inference"
        / "current_scenario_execution_id.txt"
    )

    if not execution_id_path.is_file():
        return False

    execution_id = execution_id_path.read_text(
        encoding="utf-8"
    ).strip()

    if not execution_id:
        return False

    lifecycle_paths = build_scenario_lifecycle_paths(
        base_directory=base_directory,
        execution_id=execution_id,
    )

    return lifecycle_paths.input_finished_path.is_file()


def inspect_current_scenario_drain_state(
    *,
    base_directory: Path,
) -> ScenarioDrainState | None:
    """Inspect drain state only after Slurm finishes input."""

    base_directory = Path(base_directory)

    if not current_scenario_input_is_finished(
        base_directory=base_directory
    ):
        return None

    execution_id_path = (
        base_directory
        / "inference"
        / "current_scenario_execution_id.txt"
    )
    execution_id = execution_id_path.read_text(
        encoding="utf-8"
    ).strip()
    run_started_ns = execution_id_path.stat().st_mtime_ns

    lifecycle_paths = build_scenario_lifecycle_paths(
        base_directory=base_directory,
        execution_id=execution_id,
    )

    converter_state = (
        base_directory / "state" / "pcap_to_csv"
    )
    input_agent_state = (
        base_directory / "state" / "ids_input_agent"
    )

    received_transactions: list[str] = []
    completed_transactions: list[str] = []
    pending_conversions: list[str] = []
    pending_inference: list[str] = []
    conversion_failures: list[str] = []
    inference_failures: list[str] = []

    activity_times = [
        run_started_ns,
        lifecycle_paths.input_finished_path.stat().st_mtime_ns,
    ]

    received_pcaps = sorted(
        pcap_path
        for pcap_path in (base_directory / "input").glob("*.pcap")
        if pcap_path.stat().st_mtime_ns >= run_started_ns
    )

    for pcap_path in received_pcaps:
        transaction_id = pcap_path.stem
        received_transactions.append(transaction_id)
        activity_times.append(pcap_path.stat().st_mtime_ns)

        converted_marker = (
            converter_state
            / "converted"
            / f"{pcap_path.name}.done"
        )
        conversion_failed_marker = (
            converter_state
            / "failed"
            / f"{pcap_path.name}.failed"
        )
        completed_marker = (
            input_agent_state
            / "completed"
            / f"{transaction_id}.done"
        )
        inference_failed_marker = (
            input_agent_state
            / "failed"
            / f"{transaction_id}.failed"
        )

        if conversion_failed_marker.is_file():
            conversion_failures.append(transaction_id)
            activity_times.append(
                conversion_failed_marker.stat().st_mtime_ns
            )
            continue

        if not converted_marker.is_file():
            pending_conversions.append(transaction_id)
            continue

        activity_times.append(
            converted_marker.stat().st_mtime_ns
        )

        if inference_failed_marker.is_file():
            inference_failures.append(transaction_id)
            activity_times.append(
                inference_failed_marker.stat().st_mtime_ns
            )
        elif completed_marker.is_file():
            completed_transactions.append(transaction_id)
            activity_times.append(
                completed_marker.stat().st_mtime_ns
            )
        else:
            pending_inference.append(transaction_id)

    ready_jobs = list(
        (input_agent_state / "ready").glob("*.json")
    )
    active_jobs = [
        *list(
            (input_agent_state / "active").glob("*.active")
        ),
        *list(
            (input_agent_state / "active").glob(
                "*.result.json"
            )
        ),
    ]

    for state_path in ready_jobs + active_jobs:
        activity_times.append(state_path.stat().st_mtime_ns)

    return ScenarioDrainState(
        execution_id=execution_id,
        input_finished=True,
        received_transactions=tuple(received_transactions),
        completed_transactions=tuple(
            completed_transactions
        ),
        pending_conversions=tuple(pending_conversions),
        pending_inference=tuple(pending_inference),
        conversion_failures=tuple(conversion_failures),
        inference_failures=tuple(inference_failures),
        pipeline_busy=bool(ready_jobs or active_jobs),
        latest_activity_ns=max(activity_times),
    )


def scenario_quiet_period_elapsed(
    *,
    drain_state: ScenarioDrainState,
    quiet_seconds: int = 300,
    now_ns: int | None = None,
) -> bool:
    """Return whether the drained pipeline was quiet long enough."""

    if quiet_seconds < 0:
        raise ValueError("quiet_seconds cannot be negative")

    if not drain_state.ready_for_quiet_period:
        return False

    current_time_ns = (
        time.time_ns()
        if now_ns is None
        else now_ns
    )
    required_quiet_ns = quiet_seconds * 1_000_000_000

    return (
        current_time_ns - drain_state.latest_activity_ns
        >= required_quiet_ns
    )


def _write_json_atomically(
    path: Path,
    payload: Mapping[str, object],
) -> Path:
    """Write one JSON marker using an atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(
        f"{path.name}.tmp"
    )
    temporary_path.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)

    return path


def mark_scenario_input_finished(
    *,
    base_directory: Path,
    execution_id: str,
) -> Path:
    """Record that the victim will send no more scenario PCAPs."""

    paths = build_scenario_lifecycle_paths(
        base_directory=base_directory,
        execution_id=execution_id,
    )

    return _write_json_atomically(
        paths.input_finished_path,
        {
            "status": "input_finished",
            "execution_id": execution_id,
            "created_at_utc": _utc_timestamp(),
        },
    )


def mark_scenario_complete(
    *,
    base_directory: Path,
    execution_id: str,
    summary: Mapping[str, object],
) -> Path:
    """Record the final IDS processing summary."""

    paths = build_scenario_lifecycle_paths(
        base_directory=base_directory,
        execution_id=execution_id,
    )

    payload = {
        "status": "complete",
        "execution_id": execution_id,
        "completed_at_utc": _utc_timestamp(),
        **dict(summary),
    }

    return _write_json_atomically(
        paths.complete_path,
        payload,
    )
