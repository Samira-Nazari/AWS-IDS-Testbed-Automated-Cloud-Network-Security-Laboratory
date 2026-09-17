"""Prepare completed IDS CSV files and publish inference queue jobs."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aws_ids_testbed_10.ids_inference_input import (
    prepare_inference_input,
)
from aws_ids_testbed_10.ids_inference_queue import (
    publish_inference_queue_job,
    reserve_inference_queue_job,
)
from aws_ids_testbed_10.ids_inference_workspace import (
    InferenceWorkspacePaths,
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)
from aws_ids_testbed_10.ids_scenario_lifecycle import (
    inspect_current_scenario_drain_state,
    mark_scenario_complete,
    scenario_quiet_period_elapsed,
)
from aws_ids_testbed_10.ids_stream_buffer_store import (
    commit_successful_buffer,
    save_waiting_buffer,
)


_CONVERTED_MARKER_SUFFIX = ".pcap.done"

_CAPTURE_NAME_PATTERN = re.compile(
    r"^(?P<scenario>.+)_[0-9]{8}_[0-9]{6}$"
)


@dataclass(frozen=True)
class InputCandidate:
    """One completed CSV selected for IDS preparation."""

    scenario: str
    transaction_id: str
    converted_marker_path: Path
    csv_path: Path


PreparationStatus = Literal["waiting", "queued"]


@dataclass(frozen=True)
class InputPreparationResult:
    """Result of preparing one completed converter CSV."""

    status: PreparationStatus
    transaction_id: str
    combined_rows: int
    buffer_rows: int
    inference_csv_path: Path | None
    ready_job_path: Path | None


def _transaction_id_from_marker(marker_path: Path) -> str:
    """Derive the original PCAP/CSV stem."""

    marker_name = marker_path.name

    if not marker_name.endswith(_CONVERTED_MARKER_SUFFIX):
        raise ValueError(
            f"Unsupported converter marker: {marker_path}"
        )

    return marker_name[: -len(_CONVERTED_MARKER_SUFFIX)]


def _scenario_from_transaction_id(
    transaction_id: str,
) -> str:
    """Extract the scenario without hard-coding scenario names."""

    match = _CAPTURE_NAME_PATTERN.fullmatch(transaction_id)

    if match is None:
        return "unknown"

    return match.group("scenario")


def _already_handled(
    *,
    transaction_id: str,
    state_directory: Path,
) -> bool:
    """Return whether this source CSV is queued or finalized."""

    state_markers = (
        state_directory
        / "queued"
        / f"{transaction_id}.json",
        state_directory
        / "active"
        / f"{transaction_id}.active",
        state_directory
        / "completed"
        / f"{transaction_id}.done",
        state_directory
        / "failed"
        / f"{transaction_id}.failed",
    )

    return any(marker.exists() for marker in state_markers)


def find_next_completed_csv(
    *,
    converted_marker_directory: Path,
    csv_directory: Path,
    state_directory: Path,
) -> InputCandidate | None:
    """Select the next usable completed CSV of any traffic type."""

    converted_marker_directory = Path(
        converted_marker_directory
    )
    csv_directory = Path(csv_directory)
    state_directory = Path(state_directory)

    if not converted_marker_directory.is_dir():
        return None

    converted_markers = sorted(
        (
            path
            for path in converted_marker_directory.iterdir()
            if path.is_file()
            and path.name.endswith(_CONVERTED_MARKER_SUFFIX)
        ),
        key=lambda path: (
            path.stat().st_mtime_ns,
            path.name,
        ),
    )

    for converted_marker_path in converted_markers:
        transaction_id = _transaction_id_from_marker(
            converted_marker_path
        )

        if _already_handled(
            transaction_id=transaction_id,
            state_directory=state_directory,
        ):
            continue

        csv_path = csv_directory / f"{transaction_id}.csv"

        # Missing or empty CSVs do not block later files.
        if not csv_path.is_file():
            continue

        if csv_path.stat().st_size == 0:
            continue

        return InputCandidate(
            scenario=_scenario_from_transaction_id(
                transaction_id
            ),
            transaction_id=transaction_id,
            converted_marker_path=converted_marker_path,
            csv_path=csv_path,
        )

    return None


def create_input_agent_state_directories(
    state_directory: Path,
) -> None:
    """Create the shared producer/consumer state directories."""

    for state_name in (
        "queued",
        "active",
        "completed",
        "failed",
    ):
        (Path(state_directory) / state_name).mkdir(
            parents=True,
            exist_ok=True,
        )


def _write_json_atomically(
    path: Path,
    payload: dict,
) -> None:
    """Write one state marker atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")

    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _planning_csv_path(
    *,
    workspace_paths: InferenceWorkspacePaths,
    candidate: InputCandidate,
) -> Path:
    """Return an Agent-1-only hidden preparation path."""

    return workspace_paths.queue_directory / (
        f".{candidate.transaction_id}.preparing.csv"
    )


def prepare_selected_csv(
    *,
    candidate: InputCandidate,
    workspace_paths: InferenceWorkspacePaths,
    state_directory: Path,
) -> InputPreparationResult:
    """Advance continuity and queue one source without waiting."""

    state_directory = Path(state_directory)
    create_input_agent_state_directories(state_directory)
    create_inference_workspace_directories(workspace_paths)

    completed_path = (
        state_directory
        / "completed"
        / f"{candidate.transaction_id}.done"
    )
    queued_path = (
        state_directory
        / "queued"
        / f"{candidate.transaction_id}.json"
    )
    planning_path = _planning_csv_path(
        workspace_paths=workspace_paths,
        candidate=candidate,
    )

    prepared = prepare_inference_input(
        new_csv_path=candidate.csv_path,
        buffer_csv_path=workspace_paths.buffer_csv_path,
        output_path=planning_path,
    )

    if not prepared.plan.can_run_inference:
        buffer_state = save_waiting_buffer(
            new_csv_path=candidate.csv_path,
            existing_buffer_path=(
                workspace_paths.buffer_csv_path
            ),
            buffer_output_path=(
                workspace_paths.buffer_csv_path
            ),
        )

        _write_json_atomically(
            completed_path,
            {
                "status": "buffered",
                "scenario": candidate.scenario,
                "transaction_id": candidate.transaction_id,
                "combined_rows": prepared.total_rows,
                "buffer_rows": buffer_state.row_count,
            },
        )
        candidate.csv_path.unlink()

        return InputPreparationResult(
            status="waiting",
            transaction_id=candidate.transaction_id,
            combined_rows=prepared.total_rows,
            buffer_rows=buffer_state.row_count,
            inference_csv_path=None,
            ready_job_path=None,
        )

    reservation = reserve_inference_queue_job(
        queue_directory=workspace_paths.queue_directory,
        scenario=candidate.scenario,
        transaction_id=candidate.transaction_id,
    )
    planning_path.replace(reservation.staging_csv_path)

    # Agent 1 owns continuity. Advance it before preparing the next CSV;
    # Agent 2 never modifies or rolls back this buffer.
    buffer_state = commit_successful_buffer(
        inference_input_path=reservation.staging_csv_path,
        buffer_output_path=workspace_paths.buffer_csv_path,
        plan=prepared.plan,
    )

    _write_json_atomically(
        queued_path,
        {
            "status": "queued",
            "scenario": candidate.scenario,
            "transaction_id": candidate.transaction_id,
            "sequence": reservation.sequence,
        },
    )

    try:
        job_paths = publish_inference_queue_job(
            reservation=reservation,
            metadata={
                "source_csv_path": str(candidate.csv_path),
                "buffer_rows": prepared.buffer_rows,
                "new_rows": prepared.new_rows,
                "total_rows": prepared.total_rows,
                "window_size": prepared.plan.window_size,
                "stride": prepared.plan.stride,
                "window_count": prepared.plan.window_count,
                "next_start_offset": (
                    prepared.plan.next_start_offset
                ),
                "retained_rows": prepared.plan.retained_rows,
            },
        )
    except Exception:
        queued_path.unlink(missing_ok=True)
        raise

    candidate.csv_path.unlink()

    return InputPreparationResult(
        status="queued",
        transaction_id=candidate.transaction_id,
        combined_rows=prepared.total_rows,
        buffer_rows=buffer_state.row_count,
        inference_csv_path=job_paths.csv_path,
        ready_job_path=job_paths.manifest_path,
    )


def run_input_agent_once(
    *,
    base_directory: Path,
    converted_marker_directory: Path,
    csv_directory: Path,
    state_directory: Path,
) -> InputPreparationResult | None:
    """Prepare and enqueue one completed converter CSV."""

    state_directory = Path(state_directory)
    create_input_agent_state_directories(state_directory)

    candidate = find_next_completed_csv(
        converted_marker_directory=(
            converted_marker_directory
        ),
        csv_directory=csv_directory,
        state_directory=state_directory,
    )

    if candidate is None:
        return None

    workspace_paths = build_inference_workspace_paths(
        base_directory=base_directory,
        scenario=candidate.scenario,
    )
    create_inference_workspace_directories(workspace_paths)

    return prepare_selected_csv(
        candidate=candidate,
        workspace_paths=workspace_paths,
        state_directory=state_directory,
    )


def run_input_agent_forever(
    *,
    base_directory: Path,
    converted_marker_directory: Path,
    csv_directory: Path,
    state_directory: Path,
    poll_seconds: float = 2.0,
    quiet_seconds: int = 300,
) -> None:
    """Continuously prepare completed CSVs without waiting for Agent 2."""

    print("[ids-input-agent] Started.", flush=True)

    while True:
        result = run_input_agent_once(
            base_directory=base_directory,
            converted_marker_directory=(
                converted_marker_directory
            ),
            csv_directory=csv_directory,
            state_directory=state_directory,
        )

        if result is not None:
            print(
                "[ids-input-agent] "
                f"transaction={result.transaction_id} "
                f"status={result.status} "
                f"buffer_rows={result.buffer_rows}",
                flush=True,
            )
            continue

        drain_state = inspect_current_scenario_drain_state(
            base_directory=base_directory,
        )

        if (
            drain_state is not None
            and scenario_quiet_period_elapsed(
                drain_state=drain_state,
                quiet_seconds=quiet_seconds,
            )
        ):
            complete_path = mark_scenario_complete(
                base_directory=base_directory,
                execution_id=drain_state.execution_id,
                summary={
                    "received_pcap_count": len(
                        drain_state.received_transactions
                    ),
                    "completed_transaction_count": len(
                        drain_state.completed_transactions
                    ),
                    "conversion_failure_count": len(
                        drain_state.conversion_failures
                    ),
                    "inference_failure_count": len(
                        drain_state.inference_failures
                    ),
                    "conversion_failures": (
                        drain_state.conversion_failures
                    ),
                    "inference_failures": (
                        drain_state.inference_failures
                    ),
                },
            )

            print(
                "[ids-input-agent] Scenario complete: "
                f"{complete_path}",
                flush=True,
            )
            return

        time.sleep(poll_seconds)


def main() -> None:
    """Run Agent 1 using AWS IDS runtime paths."""

    base_directory = Path(
        os.environ.get(
            "IDS_BASE_DIR",
            "/home/ubuntu/aws_ids_testbed",
        )
    )

    run_input_agent_forever(
        base_directory=base_directory,
        converted_marker_directory=(
            base_directory
            / "state"
            / "pcap_to_csv"
            / "converted"
        ),
        csv_directory=base_directory / "output" / "csv",
        state_directory=(
            base_directory / "state" / "ids_input_agent"
        ),
        poll_seconds=float(
            os.environ.get("IDS_AGENT_POLL_SECONDS", "2")
        ),
        quiet_seconds=int(
            os.environ.get(
                "IDS_SCENARIO_QUIET_SECONDS",
                "300",
            )
        ),
    )


if __name__ == "__main__":
    main()
