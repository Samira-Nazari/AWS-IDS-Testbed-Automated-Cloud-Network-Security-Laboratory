"""Calculate scenario-level IDS prediction metrics."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


GROUND_TRUTH_BY_SCENARIO = {
    "benign_http": "Benign_Final",
    "benign_iot_5_generators": "Benign_Final",
    "dos_http_flood": "DoS-HTTP_Flood",
    "dos_syn_flood": "DoS-SYN_Flood",
}


DEFAULT_METRICS_PATH = (
    "artifacts/ids_inference/ids_metrics.jsonl"
)


def calculate_prediction_metrics(
    *,
    prediction_csv_path: Path,
    scenario: str,
    scenario_execution_id: str,
) -> dict[str, object]:
    """Calculate metrics for one scenario execution only."""

    ground_truth_label = GROUND_TRUTH_BY_SCENARIO.get(scenario)

    if ground_truth_label is None:
        raise ValueError(
            f"Unsupported scenario for metrics: {scenario}"
        )

    with Path(prediction_csv_path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as prediction_file:
        rows = list(csv.DictReader(prediction_file))

    current_rows = [
        row
        for row in rows
        if row.get("scenario_execution_id")
        == scenario_execution_id
    ]

    if not current_rows:
        return {
            "scenario": scenario,
            "scenario_execution_id": scenario_execution_id,
            "ground_truth_label": ground_truth_label,
            "window_count": 0,
            "accuracy": 0.0,
            "precision_for_ground_truth_class": 0.0,
            "recall_for_ground_truth_class": 0.0,
        }

    predicted_labels = [
        row.get("predicted_label_name", "")
        for row in current_rows
    ]

    true_positives = sum(
        predicted_label == ground_truth_label
        for predicted_label in predicted_labels
    )
    false_negatives = len(predicted_labels) - true_positives

    accuracy = true_positives / len(predicted_labels)

    precision = (
        true_positives / (true_positives + 0)
        if true_positives > 0
        else 0.0
    )

    recall = (
        true_positives
        / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    return {
        "scenario": scenario,
        "scenario_execution_id": scenario_execution_id,
        "ground_truth_label": ground_truth_label,
        "window_count": len(predicted_labels),
        "accuracy": round(accuracy, 6),
        "precision_for_ground_truth_class": round(
            precision,
            6,
        ),
        "recall_for_ground_truth_class": round(
            recall,
            6,
        ),
    }


def append_prediction_metrics(
    *,
    metrics: dict[str, object],
    output_path: Path,
) -> Path:
    """Append one scenario metrics record to JSONL."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **metrics,
    }

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(record, metrics_file)
        metrics_file.write("\n")

    return output_path
