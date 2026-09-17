"""Consume queued IDS inference jobs in sequence."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from aws_ids_testbed_10.ids_inference_pipeline import (
    run_inference_steps_once,
)
from aws_ids_testbed_10.ids_inference_scenario import (
    read_scenario_execution_id,
)
from aws_ids_testbed_10.ids_master_predictions import (
    update_master_predictions,
)
from aws_ids_testbed_10.ids_scenario_lifecycle import (
    build_scenario_lifecycle_paths,
)

from aws_ids_testbed_10.ids_inference_workspace import (
    build_inference_workspace_paths,
    create_inference_workspace_directories,
    validate_runtime_name,
)


_REQUIRED_MANIFEST_FIELDS = {
    "status",
    "sequence",
    "scenario",
    "transaction_id",
    "queue_csv_path",
}


InferenceJobStatus = Literal["complete", "failed"]


@dataclass(frozen=True)
class InferenceAgentResult:
    """Final result of one inference queue job."""

    status: InferenceJobStatus
    transaction_id: str
    attempts: int
    prediction_rows: int
    master_prediction_rows: int
    result_path: Path
    error: str | None


def read_queue_manifest(
    manifest_path: Path,
) -> dict:
    """Read and validate one inference queue manifest."""

    manifest_path = Path(manifest_path)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    missing_fields = (
        _REQUIRED_MANIFEST_FIELDS - manifest.keys()
    )

    if missing_fields:
        raise ValueError(
            "Queue manifest is missing: "
            + ", ".join(sorted(missing_fields))
        )

    if manifest["status"] != "ready":
        raise ValueError(
            f"Unsupported queue status: {manifest['status']}"
        )

    sequence = int(manifest["sequence"])

    if sequence <= 0:
        raise ValueError(
            "Queue sequence must be positive"
        )

    scenario = validate_runtime_name(
        str(manifest["scenario"]),
        field_name="scenario",
    )
    transaction_id = validate_runtime_name(
        str(manifest["transaction_id"]),
        field_name="transaction_id",
    )

    queue_csv_path = Path(
        str(manifest["queue_csv_path"])
    )

    if queue_csv_path.parent != manifest_path.parent:
        raise ValueError(
            "Queued CSV and manifest must use the same directory"
        )

    if queue_csv_path.stem != manifest_path.stem:
        raise ValueError(
            "Queued CSV and manifest names do not match"
        )

    if not queue_csv_path.is_file():
        raise FileNotFoundError(
            f"Queued CSV does not exist: {queue_csv_path}"
        )

    manifest["sequence"] = sequence
    manifest["scenario"] = scenario
    manifest["transaction_id"] = transaction_id
    manifest["queue_csv_path"] = str(queue_csv_path)

    return manifest


def find_next_ready_job(
    queue_directory: Path,
) -> Path | None:
    """Return the queued manifest with the smallest sequence."""

    queue_directory = Path(queue_directory)

    if not queue_directory.is_dir():
        return None

    ready_jobs: list[tuple[int, str, Path]] = []

    for manifest_path in queue_directory.glob("*.json"):
        manifest = read_queue_manifest(manifest_path)

        ready_jobs.append(
            (
                int(manifest["sequence"]),
                manifest_path.name,
                manifest_path,
            )
        )

    if not ready_jobs:
        return None

    ready_jobs.sort()

    return ready_jobs[0][2]


def _current_scenario_is_complete(
    base_directory: Path,
) -> bool:
    """Return whether the current scenario has been fully drained."""

    workspace = build_inference_workspace_paths(
        base_directory=base_directory,
    )
    execution_id_path = workspace.current_scenario_execution_id_path

    if not execution_id_path.is_file():
        return False

    execution_id = read_scenario_execution_id(workspace)
    lifecycle_paths = build_scenario_lifecycle_paths(
        base_directory=base_directory,
        execution_id=execution_id,
    )
    return lifecycle_paths.complete_path.is_file()


def _write_json_atomically(
    path: Path,
    payload: dict,
) -> None:
    """Write one Agent 2 state marker atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _clean_finished_job(
    *,
    transaction_id: str,
    active_csv_path: Path,
    active_manifest_path: Path,
    active_marker_path: Path,
    state_directory: Path,
) -> None:
    """Remove temporary files after final success or failure."""

    active_csv_path.unlink(missing_ok=True)
    active_manifest_path.unlink(missing_ok=True)
    active_marker_path.unlink(missing_ok=True)

    (
        Path(state_directory)
        / "queued"
        / f"{transaction_id}.json"
    ).unlink(missing_ok=True)


def execute_ready_job(
    *,
    ready_job_path: Path,
    base_directory: Path,
    state_directory: Path,
) -> InferenceAgentResult:
    """Claim one queued job, run Steps 1-4, and finalize it."""

    ready_job_path = Path(ready_job_path)
    base_directory = Path(base_directory)
    state_directory = Path(state_directory)
    manifest = read_queue_manifest(ready_job_path)

    scenario = str(manifest["scenario"])
    transaction_id = str(manifest["transaction_id"])
    queue_csv_path = Path(str(manifest["queue_csv_path"]))

    active_directory = state_directory / "active"
    completed_directory = state_directory / "completed"
    failed_directory = state_directory / "failed"

    for directory in (
        active_directory,
        completed_directory,
        failed_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if any(active_directory.glob("*.active")):
        raise RuntimeError(
            "Another inference transaction is active"
        )

    workspace = build_inference_workspace_paths(
        base_directory=base_directory,
        scenario=scenario,
    )
    create_inference_workspace_directories(workspace)

    # Agent 2 is the only process that writes the shared Step 1 input.
    for old_input in workspace.input_directory.glob(
        "*_inference.csv"
    ):
        old_input.unlink()

    active_csv_path = (
        workspace.input_directory / queue_csv_path.name
    )
    active_manifest_path = (
        active_directory / f"{transaction_id}.job.json"
    )
    active_marker_path = (
        active_directory / f"{transaction_id}.active"
    )

    # Moving the manifest removes the job from the ready queue.
    ready_job_path.replace(active_manifest_path)
    queue_csv_path.replace(active_csv_path)

    _write_json_atomically(
        active_marker_path,
        {
            "status": "active",
            "sequence": manifest["sequence"],
            "scenario": scenario,
            "transaction_id": transaction_id,
            "active_csv_path": str(active_csv_path),
        },
    )

    workspace = replace(
        workspace,
        inference_csv_path=active_csv_path,
    )
    execution_id = read_scenario_execution_id(workspace)
    last_error: str | None = None

    for attempt in range(1, 3):
        try:
            pipeline_result = run_inference_steps_once(
                paths=workspace,
            )

            master_result = update_master_predictions(
                current_predictions_path=(
                    pipeline_result.predictions_path
                ),
                master_predictions_path=(
                    workspace.master_predictions_path
                ),
                scenario=scenario,
                scenario_execution_id=execution_id,
                transaction_id=transaction_id,
                inference_csv_name=active_csv_path.name,
            )

            result_path = (
                completed_directory
                / f"{transaction_id}.done"
            )

            _write_json_atomically(
                result_path,
                {
                    "status": "complete",
                    "sequence": manifest["sequence"],
                    "scenario": scenario,
                    "transaction_id": transaction_id,
                    "attempts": attempt,
                    "prediction_rows": (
                        pipeline_result.prediction_count
                    ),
                    "master_prediction_rows": (
                        master_result.total_rows
                    ),
                },
            )

            _clean_finished_job(
                transaction_id=transaction_id,
                active_csv_path=active_csv_path,
                active_manifest_path=active_manifest_path,
                active_marker_path=active_marker_path,
                state_directory=state_directory,
            )

            return InferenceAgentResult(
                status="complete",
                transaction_id=transaction_id,
                attempts=attempt,
                prediction_rows=(
                    pipeline_result.prediction_count
                ),
                master_prediction_rows=(
                    master_result.total_rows
                ),
                result_path=result_path,
                error=None,
            )

        except Exception as exception:
            last_error = (
                f"{type(exception).__name__}: {exception}"
            )

    result_path = (
        failed_directory / f"{transaction_id}.failed"
    )

    _write_json_atomically(
        result_path,
        {
            "status": "failed",
            "sequence": manifest["sequence"],
            "scenario": scenario,
            "transaction_id": transaction_id,
            "attempts": 2,
            "prediction_rows": 0,
            "master_prediction_rows": 0,
            "error": last_error,
        },
    )

    _clean_finished_job(
        transaction_id=transaction_id,
        active_csv_path=active_csv_path,
        active_manifest_path=active_manifest_path,
        active_marker_path=active_marker_path,
        state_directory=state_directory,
    )

    return InferenceAgentResult(
        status="failed",
        transaction_id=transaction_id,
        attempts=2,
        prediction_rows=0,
        master_prediction_rows=0,
        result_path=result_path,
        error=last_error,
    )


def run_inference_agent_once(
    *,
    base_directory: Path,
    state_directory: Path,
) -> InferenceAgentResult | None:
    """Execute the next queued inference job, if available."""

    workspace = build_inference_workspace_paths(
        base_directory=base_directory,
    )
    ready_job_path = find_next_ready_job(
        workspace.queue_directory
    )

    if ready_job_path is None:
        return None

    return execute_ready_job(
        ready_job_path=ready_job_path,
        base_directory=base_directory,
        state_directory=state_directory,
    )


def run_inference_agent_forever(
    *,
    base_directory: Path,
    state_directory: Path,
    poll_seconds: float = 2.0,
) -> None:
    """Continuously consume inference jobs in queue order."""

    print("[ids-inference-agent] Started.", flush=True)

    while True:
        result = run_inference_agent_once(
            base_directory=base_directory,
            state_directory=state_directory,
        )

        if result is not None:
            print(
                "[ids-inference-agent] "
                f"transaction={result.transaction_id} "
                f"status={result.status} "
                f"attempts={result.attempts} "
                f"predictions={result.prediction_rows}",
                flush=True,
            )
            continue

        if _current_scenario_is_complete(base_directory):
            print(
                "[ids-inference-agent] Scenario complete. Exiting.",
                flush=True,
            )
            return

        time.sleep(poll_seconds)


def main() -> None:
    """Run Agent 2 using AWS IDS runtime paths."""

    base_directory = Path(
        os.environ.get(
            "IDS_BASE_DIR",
            "/home/ubuntu/aws_ids_testbed",
        )
    )

    run_inference_agent_forever(
        base_directory=base_directory,
        state_directory=(
            base_directory / "state" / "ids_input_agent"
        ),
        poll_seconds=float(
            os.environ.get("IDS_AGENT_POLL_SECONDS", "2")
        ),
    )


if __name__ == "__main__":
    main()
