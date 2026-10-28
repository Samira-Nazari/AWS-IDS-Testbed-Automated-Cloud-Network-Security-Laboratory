"""Write one structured alert event for each successful IDS window."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _local_timestamp() -> str:
    """Return a readable timestamp in the configured local timezone."""

    timezone_name = os.environ.get(
        "IDS_DISPLAY_TIMEZONE",
        "America/Toronto",
    )
    local_time = datetime.now(timezone.utc).astimezone(
        ZoneInfo(timezone_name)
    )

    return local_time.strftime(
        "%B %-d, %Y at %-I:%M:%S %p"
    )


def append_prediction_events(
    *,
    predictions_path: Path,
    events_path: Path,
    scenario: str,
    scenario_execution_id: str,
    transaction_id: str,
    inference_csv_name: str,
) -> int:
    """Append one JSON event for every prediction row."""

    predictions = pd.read_csv(predictions_path)

    if predictions.empty:
        raise ValueError("Cannot write events for empty predictions")

    required_columns = {
        "predicted_label_name",
        "confidence",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            "Prediction CSV is missing: "
            + ", ".join(sorted(missing_columns))
        )

    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    diagnosed_at = _utc_timestamp()
    event_count = 0

    with events_path.open("a", encoding="utf-8") as event_file:
        for row_index, row in predictions.iterrows():
            window_index = row.get("window_index", row_index)
            label = str(row["predicted_label_name"])
            confidence = float(row["confidence"])

            event = {
                # "event_type": "ids_prediction",
                # "prediction_id": (
                #     f"{scenario_execution_id}:"
                #     f"{transaction_id}:"
                #     f"{window_index}"
                # ),
                "scenario": scenario,
                # "scenario_execution_id": scenario_execution_id,
                # "transaction_id": transaction_id,
                "inference_csv_name": inference_csv_name,
                "window_index": int(window_index),
                "predicted_label_name": label,
                "confidence": confidence,
                "verdict": (
                    "BENIGN"
                    if label == "Benign_Final"
                    else "ATTACK"
                ),
                "diagnosed_at_utc": diagnosed_at,
                "diagnosed_at_local": _local_timestamp(),
            }

            event_file.write(
                json.dumps(event, ensure_ascii=False)
                + "\n"
            )
            event_count += 1

    return event_count
