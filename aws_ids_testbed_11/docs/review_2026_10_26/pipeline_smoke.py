"""Real CPU pipeline smoke using synthetic rows and bundled model artifacts.

Run from the project root: python -B docs/review_2026_10_26/pipeline_smoke.py
All generated data and inference outputs live in a temporary directory.
"""

import csv
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

import torch
from aws_ids_testbed_11.ids_inference_pipeline import run_inference_steps_once
from aws_ids_testbed_11.ids_inference_workspace import (
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)
from ids_detector.lite_v6.config.config import (
    LITE_V6_SELECTED_FEATURES_PATH,
    LITE_V6_STANDARD_SCALER_PATH,
)

with LITE_V6_SELECTED_FEATURES_PATH.open("rb") as source:
    features = list(pickle.load(source))
with LITE_V6_STANDARD_SCALER_PATH.open("rb") as source:
    scaler = pickle.load(source)

torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as directory:
    workspace = create_inference_workspace_directories(
        build_inference_workspace_paths(
            base_directory=Path(directory), scenario="benign_http"
        )
    )
    columns = list(dict.fromkeys([*features, "Timestamp_Start"]))
    with workspace.inference_csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for index in range(35):
            row = dict(zip(features, map(float, scaler.mean_)))
            row["Timestamp_Start"] = 1700000000 + index
            writer.writerow(row)
    with patch(
        "ids_detector.lite_v6.evaluation.evaluator.IDS_INFERENCE_OUTPUT_DIR",
        Path(directory) / "evaluation",
    ):
        result = run_inference_steps_once(paths=workspace, device_setting="cpu")
    with result.predictions_path.open() as source:
        rows = list(csv.DictReader(source))
    print("REAL CPU PIPELINE: input_rows=", result.input_rows,
          "processed_rows=", result.processed_rows,
          "windows=", result.window_count,
          "prediction_count=", result.prediction_count,
          "prediction_rows=", len(rows),
          "probability_cols=", len([column for column in rows[0] if column.startswith("prob_")]))
    print("PURE MODEL:", result.pure_model_batch_count, "batch,",
          result.pure_model_window_count, "windows")
