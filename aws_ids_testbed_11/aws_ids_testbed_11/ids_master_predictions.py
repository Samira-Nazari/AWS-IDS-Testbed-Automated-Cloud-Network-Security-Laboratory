"""Maintain the cumulative IDS prediction CSV."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MasterPredictionResult:
    added_rows: int
    total_rows: int
    master_predictions_path: Path


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def update_master_predictions(
    current_predictions_path: Path,
    master_predictions_path: Path,
    scenario: str,
    scenario_execution_id: str,
    transaction_id: str,
    inference_csv_name: str,
    diagnosed_at_utc: str | None = None,
) -> MasterPredictionResult:
    """Add one successful transaction to the cumulative prediction CSV."""

    current_predictions_path = Path(current_predictions_path)
    master_predictions_path = Path(master_predictions_path)

    current = pd.read_csv(current_predictions_path)

    if current.empty:
        raise ValueError(
            f"Current prediction file is empty: {current_predictions_path}"
        )

    if "window_index" in current.columns:
        window_identifiers = current["window_index"].astype(str)
    else:
        window_identifiers = pd.Series(
            range(len(current)),
            index=current.index,
            dtype=str,
        )

    current["prediction_id"] = (
        scenario_execution_id
        + ":"
        + transaction_id
        + ":"
        + window_identifiers
    )
    current["scenario"] = scenario
    current["scenario_execution_id"] = scenario_execution_id
    current["transaction_id"] = transaction_id
    current["inference_csv_name"] = inference_csv_name
    current["diagnosed_at_utc"] = diagnosed_at_utc or _utc_timestamp()

    metadata_columns = [
        "prediction_id",
        "scenario",
        "scenario_execution_id",
        "transaction_id",
        "inference_csv_name",
        "diagnosed_at_utc",
    ]
    remaining_columns = [
        column
        for column in current.columns
        if column not in metadata_columns
    ]
    current = current[metadata_columns + remaining_columns]

    if master_predictions_path.exists():
        master = pd.read_csv(master_predictions_path)

        if "prediction_id" not in master.columns:
            raise ValueError(
                "Existing master prediction CSV has no prediction_id column"
            )
    else:
        master = pd.DataFrame()

    previous_total = len(master)

    combined = pd.concat(
        [master, current],
        ignore_index=True,
        sort=False,
    )
    combined = combined.drop_duplicates(
        subset=["prediction_id"],
        keep="first",
    )

    master_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = master_predictions_path.with_suffix(".csv.tmp")
    combined.to_csv(temporary_path, index=False)
    temporary_path.replace(master_predictions_path)

    return MasterPredictionResult(
        added_rows=len(combined) - previous_total,
        total_rows=len(combined),
        master_predictions_path=master_predictions_path,
    )
