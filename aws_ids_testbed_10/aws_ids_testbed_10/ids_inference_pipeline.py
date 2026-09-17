"""Run the existing IDS inference Steps 1–4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ids_detector.lite_v6.config.config import (
    IDS_PREDICTION_BATCH_SIZE,
    IDS_PREDICTION_DEVICE,
)
from ids_detector.lite_v6.data.data_loader import (
    load_data as run_step_1,
)
from ids_detector.lite_v6.preprocessing.preprocessor import (
    run_step_2,
)
from ids_detector.lite_v6.features.feature_engineering import (
    run_step_3,
)
from ids_detector.lite_v6.evaluation.evaluator import (
    run_step_4,
)

from aws_ids_testbed_10.ids_inference_workspace import (
    InferenceWorkspacePaths,
)


@dataclass(frozen=True)
class InferencePipelineResult:
    """Result of one successful Steps 1–4 execution."""

    input_rows: int
    processed_rows: int
    window_count: int
    prediction_count: int
    predictions_path: Path


def run_inference_steps_once(
    *,
    paths: InferenceWorkspacePaths,
    batch_size: int = IDS_PREDICTION_BATCH_SIZE,
    device_setting: str = IDS_PREDICTION_DEVICE,
) -> InferencePipelineResult:
    """Call the existing Step 1, 2, 3 and 4 functions."""

    if paths.inference_csv_path is None:
        raise ValueError("Inference CSV path is not configured")

    if not paths.inference_csv_path.exists():
        raise FileNotFoundError(
            f"Inference CSV not found: {paths.inference_csv_path}"
        )

    manifest_path = (
        paths.inference_directory / "ids_capture_manifest.csv"
    )

    step_1_data = run_step_1(
        input_dir=paths.input_directory,
        output_path=paths.step_1_path,
        manifest_path=manifest_path,
    )

    step_2_data = run_step_2(
        input_path=paths.step_1_path,
        output_path=paths.step_2_path,
    )

    step_3_result = run_step_3(
        input_path=paths.step_2_path,
        window_output_path=paths.windows_path,
        metadata_output_path=paths.window_metadata_path,
        features_output_path=paths.selected_features_path,
        summary_output_path=paths.windowing_summary_path,
    )

    step_4_result = run_step_4(
        window_data_path=paths.windows_path,
        metadata_path=paths.window_metadata_path,
        predictions_output_path=paths.current_predictions_path,
        summary_output_path=paths.prediction_summary_path,
        batch_size=batch_size,
        device_setting=device_setting,
    )

    return InferencePipelineResult(
        input_rows=len(step_1_data),
        processed_rows=len(step_2_data),
        window_count=len(step_3_result["X_windows"]),
        prediction_count=int(
            step_4_result["summary"]["prediction_rows"]
        ),
        predictions_path=paths.current_predictions_path,
    )
