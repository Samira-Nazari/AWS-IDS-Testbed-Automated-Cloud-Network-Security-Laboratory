"""Watch the master IDS prediction CSV and create alert events."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from aws_ids_testbed_11.ids_timing import (
    append_timing_event,
    utc_now_iso,
)


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 form."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def local_now() -> str:
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


def load_seen_ids(
    state_path: Path,
    events_path: Path,
) -> set[str]:
    """Load processed IDs, recovering from an existing event log."""

    seen: set[str] = set()

    if state_path.exists():
        payload = json.loads(
            state_path.read_text(encoding="utf-8")
        )
        seen.update(
            str(value)
            for value in payload.get("prediction_ids", [])
        )
        return seen

    if events_path.exists():
        for line in events_path.read_text(
            encoding="utf-8"
        ).splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            prediction_id = event.get("prediction_id")

            if prediction_id:
                seen.add(str(prediction_id))

    return seen


def save_seen_ids(
    state_path: Path,
    seen_ids: set[str],
) -> None:
    """Persist processed IDs atomically."""

    state_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = state_path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(
            {
                "prediction_ids": sorted(seen_ids),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(state_path)


def _timestamp_delta_ms(
    started_at_utc: object,
    completed_at_utc: str,
) -> float | None:
    """Calculate elapsed milliseconds between UTC ISO timestamps."""

    if not started_at_utc:
        return None

    try:
        started = datetime.fromisoformat(
            str(started_at_utc).replace("Z", "+00:00")
        )
        completed = datetime.fromisoformat(
            completed_at_utc.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    return round((completed - started).total_seconds() * 1000, 3)


def _read_timing_events(
    timing_events_path: Path,
) -> list[dict]:
    """Read timing events needed to correlate alert latency."""

    if not timing_events_path.is_file():
        return []

    events: list[dict] = []
    for line in timing_events_path.read_text(
        encoding="utf-8"
    ).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(event, dict):
            events.append(event)

    return events


def _append_timing_event_safely(
    *,
    timing_events_path: Path,
    event: dict[str, object],
) -> None:
    """Record alert timing without changing alert processing behavior."""

    try:
        append_timing_event(timing_events_path, event)
    except Exception as exception:
        print(
            "[ids-alert-agent] Timing event could not be recorded: "
            f"{type(exception).__name__}: {exception}",
            flush=True,
        )


def process_new_predictions(
    *,
    predictions_path: Path,
    events_path: Path,
    state_path: Path,
) -> int:
    """Append one event for each unseen master prediction row."""

    if not predictions_path.is_file():
        return 0

    predictions = pd.read_csv(predictions_path)

    if predictions.empty:
        return 0

    required_columns = {
        "prediction_id",
        "predicted_label_name",
        "confidence",
    }
    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            "Prediction CSV is missing: "
            + ", ".join(sorted(missing))
        )

    seen_ids = load_seen_ids(
        state_path=state_path,
        events_path=events_path,
    )

    new_predictions = predictions[
        ~predictions["prediction_id"]
        .astype(str)
        .isin(seen_ids)
    ]

    if new_predictions.empty:
        return 0

    events_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timing_events_file = events_path.parent / "ids_timing_events.jsonl"
    inference_timing_events = _read_timing_events(timing_events_file)
    inference_events_by_transaction = {
        str(event.get("transaction_id")): event
        for event in inference_timing_events
        if event.get("event_type") == "inference_timing"
        and event.get("transaction_id")
    }
    inference_events_by_csv = {
        str(event.get("inference_csv_name")): event
        for event in inference_timing_events
        if event.get("event_type") == "inference_timing"
        and event.get("inference_csv_name")
    }
    capture_events_by_id = {
        str(event.get("capture_id")): event
        for event in inference_timing_events
        if event.get("event_type") == "pcap_received"
        and event.get("capture_id")
    }

    alert_processing_started_at_utc = utc_now_iso()
    alert_processing_started_monotonic_ns = time.monotonic_ns()
    alert_groups: dict[str, dict[str, object]] = {}
    added = 0

    with events_path.open("a", encoding="utf-8") as event_file:
        for _, row in new_predictions.iterrows():
            prediction_id = str(row["prediction_id"])
            label = str(row["predicted_label_name"])
            confidence = float(row["confidence"])
            verdict = (
                "BENIGN"
                if label == "Benign_Final"
                else "ATTACK"
            )

            event = {
                # "event_type": "ids_prediction",
                # "prediction_id": prediction_id,
                "scenario": str(row.get("scenario", "")),
                # "scenario_execution_id": str(
                #     row.get("scenario_execution_id", "")
                # ),
                # "transaction_id": str(
                #     row.get("transaction_id", "")
                # ),
                "inference_csv_name": str(
                    row.get("inference_csv_name", "")
                ),
                "window_index": int(
                    row.get("window_index", 0)
                ),
                "predicted_label_name": label,
                "confidence": confidence,
                "verdict": verdict,
                "detected_at_utc": utc_now(),
                "detected_at_local": local_now(),
            }

            event_file.write(
                json.dumps(event, ensure_ascii=False)
                + "\n"
            )

            transaction_id_value = row.get("transaction_id", "")
            if pd.isna(transaction_id_value):
                transaction_id_value = ""
            transaction_id = str(transaction_id_value)

            inference_csv_value = row.get(
                "inference_csv_name",
                "",
            )
            if pd.isna(inference_csv_value):
                inference_csv_value = ""
            inference_csv_name = str(inference_csv_value)

            scenario_value = row.get("scenario", "")
            if pd.isna(scenario_value):
                scenario_value = ""
            scenario = str(scenario_value)

            group_key = transaction_id or inference_csv_name
            group = alert_groups.setdefault(
                group_key,
                {
                    "scenario": scenario,
                    "transaction_id": transaction_id,
                    "inference_csv_name": inference_csv_name,
                    "prediction_count": 0,
                },
            )
            group["prediction_count"] = int(
                group["prediction_count"]
            ) + 1

            seen_ids.add(prediction_id)
            added += 1

            print(
                "[ids-alert-agent] "
                f"{verdict} "
                f"label={label} "
                f"confidence={confidence:.4f}",
                flush=True,
            )

    alert_recorded_at_utc = utc_now_iso()
    alert_processing_duration_ms = round(
        (
            time.monotonic_ns()
            - alert_processing_started_monotonic_ns
        )
        / 1_000_000,
        3,
    )

    save_seen_ids(
        state_path=state_path,
        seen_ids=seen_ids,
    )

    for group in alert_groups.values():
        transaction_id = str(group["transaction_id"])
        inference_csv_name = str(group["inference_csv_name"])
        inference_event = (
            inference_events_by_transaction.get(transaction_id)
            or inference_events_by_csv.get(inference_csv_name)
        )
        capture_event = capture_events_by_id.get(
            transaction_id
        ) or capture_events_by_id.get(
            Path(inference_csv_name).stem.replace(
                "_inference",
                "",
            )
        )

        inference_completed_at_utc = (
            inference_event.get("inference_completed_at_utc")
            if inference_event
            else None
        )
        capture_started_at_utc = (
            capture_event.get("capture_started_at_utc")
            if capture_event
            else None
        )

        _append_timing_event_safely(
            timing_events_path=timing_events_file,
            event={
                "event_type": "alert_timing",
                "scenario": group["scenario"],
                "transaction_id": transaction_id,
                "inference_csv_name": inference_csv_name,
                "prediction_count": group["prediction_count"],
                "alert_processing_started_at_utc": (
                    alert_processing_started_at_utc
                ),
                "alert_recorded_at_utc": alert_recorded_at_utc,
                "alert_processing_duration_ms": (
                    alert_processing_duration_ms
                ),
                "inference_completed_at_utc": (
                    inference_completed_at_utc
                ),
                "inference_to_alert_latency_ms": (
                    _timestamp_delta_ms(
                        inference_completed_at_utc,
                        alert_recorded_at_utc,
                    )
                ),
                "capture_started_at_utc": capture_started_at_utc,
                "end_to_end_detection_latency_ms": (
                    _timestamp_delta_ms(
                        capture_started_at_utc,
                        alert_recorded_at_utc,
                    )
                ),
            },
        )

    return added


def run_alert_agent_forever(
    *,
    base_directory: Path,
    poll_seconds: float = 2.0,
) -> None:
    """Continuously watch the persistent prediction CSV."""

    predictions_path = (
        base_directory
        / "inference"
        / "ids_predictions.csv"
    )
    events_path = (
        base_directory
        / "logs"
        / "ids_alert_events.jsonl"
    )
    state_path = (
        base_directory
        / "state"
        / "ids_alert_agent"
        / "seen_prediction_ids.json"
    )

    print("[ids-alert-agent] Started.", flush=True)

    while True:
        process_new_predictions(
            predictions_path=predictions_path,
            events_path=events_path,
            state_path=state_path,
        )
        time.sleep(poll_seconds)


def main() -> None:
    """Run Agent 3 using AWS IDS runtime paths."""

    base_directory = Path(
        os.environ.get(
            "IDS_BASE_DIR",
            "/home/ubuntu/aws_ids_testbed",
        )
    )

    run_alert_agent_forever(
        base_directory=base_directory,
        poll_seconds=float(
            os.environ.get(
                "IDS_ALERT_AGENT_POLL_SECONDS",
                "2",
            )
        ),
    )


if __name__ == "__main__":
    main()
