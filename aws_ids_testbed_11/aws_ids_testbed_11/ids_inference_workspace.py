"""Define the shared AWS IDS inference workspace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_SAFE_RUNTIME_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
)


@dataclass(frozen=True)
class InferenceWorkspacePaths:
    """Paths in the single shared IDS inference directory."""

    base_directory: Path
    inference_directory: Path
    input_directory: Path
    queue_directory: Path

    current_scenario_path: Path
    current_scenario_execution_id_path: Path
    buffer_csv_path: Path
    inference_csv_path: Path | None

    step_1_path: Path
    step_2_path: Path
    windows_path: Path
    window_metadata_path: Path
    selected_features_path: Path
    windowing_summary_path: Path

    current_predictions_path: Path
    prediction_summary_path: Path
    master_predictions_path: Path


def validate_runtime_name(
    value: str,
    *,
    field_name: str,
) -> str:
    """Validate one name used in an IDS runtime path."""

    if not _SAFE_RUNTIME_NAME.fullmatch(value):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, "
            "periods, underscores, and hyphens"
        )

    return value


def _validate_scenario(scenario: str) -> str:
    """Validate a scenario name used in an inference filename."""

    return validate_runtime_name(
        scenario,
        field_name="scenario",
    )


def build_inference_workspace_paths(
    *,
    base_directory: Path,
    scenario: str | None = None,
) -> InferenceWorkspacePaths:
    """Build shared inference paths without creating anything."""

    base_directory = Path(base_directory)
    inference_directory = base_directory / "inference"
    input_directory = inference_directory / "input"
    queue_directory = inference_directory / "queue"

    inference_csv_path: Path | None = None

    if scenario is not None:
        scenario = _validate_scenario(scenario)
        inference_csv_path = (
            input_directory / f"{scenario}_inference.csv"
        )

    return InferenceWorkspacePaths(
        base_directory=base_directory,
        inference_directory=inference_directory,
        input_directory=input_directory,
        queue_directory=queue_directory,
        current_scenario_path=(
            inference_directory / "current_scenario.txt"
        ),
        current_scenario_execution_id_path=(
            inference_directory
            / "current_scenario_execution_id.txt"
        ),
        buffer_csv_path=(
            inference_directory / "stream_buffer.csv"
        ),
        inference_csv_path=inference_csv_path,
        step_1_path=(
            inference_directory / "step_1_Loaded_IDS_Data.pkl"
        ),
        step_2_path=(
            inference_directory / "step_2_Processed_IDS_Data.pkl"
        ),
        windows_path=(
            inference_directory / "X_IDS_windows.npy"
        ),
        window_metadata_path=(
            inference_directory / "ids_window_metadata.csv"
        ),
        selected_features_path=(
            inference_directory / "ids_selected_features_used.pkl"
        ),
        windowing_summary_path=(
            inference_directory / "ids_windowing_summary.pkl"
        ),
        current_predictions_path=(
            inference_directory / "ids_predictions_current.csv"
        ),
        prediction_summary_path=(
            inference_directory / "ids_prediction_summary.pkl"
        ),
        master_predictions_path=(
            inference_directory / "ids_predictions.csv"
        ),
    )


def create_inference_workspace_directories(
    paths: InferenceWorkspacePaths,
) -> InferenceWorkspacePaths:
    """Create the single shared inference workspace."""

    paths.input_directory.mkdir(parents=True, exist_ok=True)
    paths.queue_directory.mkdir(parents=True, exist_ok=True)

    return paths
