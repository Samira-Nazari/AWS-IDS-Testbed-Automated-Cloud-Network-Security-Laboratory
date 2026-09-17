"""Prepare the shared inference workspace for a scenario execution."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from aws_ids_testbed_10.ids_inference_workspace import (
    InferenceWorkspacePaths,
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)


def _replaceable_artifact_paths(
    paths: InferenceWorkspacePaths,
) -> tuple[Path, ...]:
    """Return files replaced during each scenario execution."""

    return (
        paths.buffer_csv_path,
        paths.step_1_path,
        paths.step_2_path,
        paths.windows_path,
        paths.window_metadata_path,
        paths.selected_features_path,
        paths.windowing_summary_path,
        paths.current_predictions_path,
        paths.prediction_summary_path,
    )


def _clear_current_inference_inputs(
    paths: InferenceWorkspacePaths,
) -> None:
    """Remove previous synthetic inference inputs."""

    if not paths.input_directory.exists():
        return

    for input_path in paths.input_directory.iterdir():
        if not input_path.is_file():
            continue

        if (
            input_path.name.endswith("_inference.csv")
            or input_path.name.endswith(".tmp")
        ):
            input_path.unlink()


def _write_current_scenario(
    paths: InferenceWorkspacePaths,
    scenario: str,
) -> None:
    """Atomically record the active scenario."""

    temporary_path = paths.current_scenario_path.with_name(
        f"{paths.current_scenario_path.name}.tmp"
    )

    temporary_path.write_text(
        f"{scenario}\n",
        encoding="utf-8",
    )

    temporary_path.replace(paths.current_scenario_path)


def _write_scenario_execution_id(
    paths: InferenceWorkspacePaths,
    scenario_execution_id: str,
) -> None:
    """Atomically record the active scenario execution ID."""

    temporary_path = (
        paths.current_scenario_execution_id_path.with_name(
            f"{paths.current_scenario_execution_id_path.name}.tmp"
        )
    )

    temporary_path.write_text(
        f"{scenario_execution_id}\n",
        encoding="utf-8",
    )
    temporary_path.replace(
        paths.current_scenario_execution_id_path
    )


def read_scenario_execution_id(
    paths: InferenceWorkspacePaths,
) -> str:
    """Read the active scenario execution ID."""

    execution_id = (
        paths.current_scenario_execution_id_path
        .read_text(encoding="utf-8")
        .strip()
    )

    if not execution_id:
        raise ValueError("Scenario execution ID is empty")

    return execution_id


def start_inference_run(
    *,
    base_directory: Path,
    execution_id: str | None = None,
) -> InferenceWorkspacePaths:
    """Reset shared temporary inference data for a new run."""

    current_execution_id = execution_id or uuid4().hex

    paths = build_inference_workspace_paths(
        base_directory=base_directory,
    )
    create_inference_workspace_directories(paths)

    # Remove every scenario-specific inference input.
    _clear_current_inference_inputs(paths)

    # Remove the shared buffer and replaceable Steps 1-4 files.
    for artifact_path in _replaceable_artifact_paths(paths):
        artifact_path.unlink(missing_ok=True)

    # Agent 1 determines the scenario from each received CSV.
    paths.current_scenario_path.unlink(missing_ok=True)

    # Keep ids_predictions.csv and create a new run identifier.
    _write_scenario_execution_id(
        paths=paths,
        scenario_execution_id=current_execution_id,
    )

    return paths


def start_scenario_inference(
    *,
    base_directory: Path,
    scenario: str,
    scenario_execution_id: str | None = None,
) -> InferenceWorkspacePaths:
    """Reset temporary inference state for one scenario execution."""

    execution_id = scenario_execution_id or uuid4().hex

    paths = build_inference_workspace_paths(
        base_directory=base_directory,
        scenario=scenario,
    )

    create_inference_workspace_directories(paths)

    _clear_current_inference_inputs(paths)

    for artifact_path in _replaceable_artifact_paths(paths):
        artifact_path.unlink(missing_ok=True)

    _write_scenario_execution_id(
        paths=paths,
        scenario_execution_id=execution_id,
    )
    _write_current_scenario(
        paths=paths,
        scenario=scenario,
    )

    return paths
