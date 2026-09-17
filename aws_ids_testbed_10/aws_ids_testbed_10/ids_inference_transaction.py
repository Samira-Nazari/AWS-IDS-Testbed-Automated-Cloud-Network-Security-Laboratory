"""Process one received CSV as a serialized IDS transaction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aws_ids_testbed_10.ids_inference_input import (
    prepare_inference_input,
)
from aws_ids_testbed_10.ids_inference_pipeline import (
    run_inference_steps_once,
)
from aws_ids_testbed_10.ids_inference_workspace import (
    InferenceWorkspacePaths,
)
from aws_ids_testbed_10.ids_master_predictions import (
    update_master_predictions,
)
from aws_ids_testbed_10.ids_stream_buffer_store import (
    commit_successful_buffer,
    save_waiting_buffer,
)


TransactionStatus = Literal["waiting", "complete", "failed"]


@dataclass(frozen=True)
class InferenceTransactionResult:
    status: TransactionStatus
    attempts: int
    combined_rows: int
    prediction_rows: int
    master_prediction_rows: int
    buffer_rows: int
    error: str | None


def process_received_csv(
    *,
    new_csv_path: Path,
    paths: InferenceWorkspacePaths,
    scenario: str,
    scenario_execution_id: str,
    transaction_id: str,
) -> InferenceTransactionResult:
    """Process one CSV and return only after its transaction finishes."""

    if paths.inference_csv_path is None:
        raise ValueError("Inference CSV path is not configured")

    prepared = prepare_inference_input(
        new_csv_path=new_csv_path,
        buffer_csv_path=paths.buffer_csv_path,
        output_path=paths.inference_csv_path,
    )

    if not prepared.plan.can_run_inference:
        buffer_state = save_waiting_buffer(
            new_csv_path=new_csv_path,
            existing_buffer_path=paths.buffer_csv_path,
            buffer_output_path=paths.buffer_csv_path,
        )

        return InferenceTransactionResult(
            status="waiting",
            attempts=0,
            combined_rows=prepared.total_rows,
            prediction_rows=0,
            master_prediction_rows=0,
            buffer_rows=buffer_state.row_count,
            error=None,
        )

    last_error: str | None = None

    for attempt in range(1, 3):
        try:
            pipeline_result = run_inference_steps_once(
                paths=paths,
            )

            master_result = update_master_predictions(
                current_predictions_path=(
                    pipeline_result.predictions_path
                ),
                master_predictions_path=(
                    paths.master_predictions_path
                ),
                scenario=scenario,
                scenario_execution_id=scenario_execution_id,
                transaction_id=transaction_id,
                inference_csv_name=(
                    paths.inference_csv_path.name
                ),
            )

            buffer_state = commit_successful_buffer(
                inference_input_path=paths.inference_csv_path,
                buffer_output_path=paths.buffer_csv_path,
                plan=prepared.plan,
            )

            return InferenceTransactionResult(
                status="complete",
                attempts=attempt,
                combined_rows=prepared.total_rows,
                prediction_rows=pipeline_result.prediction_count,
                master_prediction_rows=master_result.total_rows,
                buffer_rows=buffer_state.row_count,
                error=None,
            )

        except Exception as exception:
            last_error = (
                f"{type(exception).__name__}: {exception}"
            )

    return InferenceTransactionResult(
        status="failed",
        attempts=2,
        combined_rows=prepared.total_rows,
        prediction_rows=0,
        master_prediction_rows=0,
        buffer_rows=prepared.buffer_rows,
        error=last_error,
    )
