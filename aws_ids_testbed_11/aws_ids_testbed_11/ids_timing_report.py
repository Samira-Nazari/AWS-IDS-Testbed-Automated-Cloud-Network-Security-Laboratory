"""Build read-only CSV timing reports from the IDS timing event log.

The runtime agents append timing events to ``logs/ids_timing_events.jsonl``.
This module only reads that log and the cumulative prediction CSV; it never
changes the capture, conversion, inference, or alert pipeline.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable


PCAP_REPORT_FIELDS = (
    "capture_id",
    "scenario",
    "scenario_execution_id",
    "pcap_filename",
    "capture_started_at_utc",
    "capture_completed_at_utc",
    "capture_duration_ms",
    "upload_started_at_utc",
    "receive_started_at_utc",
    "received_at_utc",
    "receive_duration_ms",
    "transfer_latency_ms",
    "conversion_started_at_utc",
    "conversion_completed_at_utc",
    "conversion_duration_ms",
    "conversion_status",
    "pcap_size_bytes",
    "csv_filename",
    "csv_size_bytes",
    "sha256",
)

TRANSACTION_REPORT_FIELDS = (
    "scenario",
    "scenario_execution_id",
    "transaction_id",
    "queue_sequence",
    "queued_at_utc",
    "inference_started_at_utc",
    "inference_completed_at_utc",
    "queue_wait_duration_ms",
    "total_inference_duration_ms",
    "pure_model_inference_duration_ms",
    "model_batch_count",
    "model_window_count",
    "attempts",
    "prediction_rows",
    "inference_status",
    "alert_processing_started_at_utc",
    "alert_recorded_at_utc",
    "alert_processing_duration_ms",
    "inference_to_alert_latency_ms",
    "end_to_end_detection_latency_ms",
)

WINDOW_REPORT_FIELDS = (
    "prediction_id",
    "scenario",
    "scenario_execution_id",
    "transaction_id",
    "inference_csv_name",
    "window_index",
    "predicted_label_name",
    "confidence",
    "verdict",
    "queue_wait_duration_ms",
    "total_inference_duration_ms",
    "pure_model_inference_duration_ms",
    "inference_to_alert_latency_ms",
    "end_to_end_detection_latency_ms",
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read valid JSON objects from an append-only JSONL file."""

    if not path.is_file():
        return []

    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as event_file:
        for line in event_file:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
    return events


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV while returning an empty list for a missing/empty file."""

    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _timestamp_delta_ms(
    started_at_utc: object,
    completed_at_utc: object,
) -> float | None:
    """Calculate a signed elapsed time between two ISO-8601 timestamps."""

    if not started_at_utc or not completed_at_utc:
        return None
    try:
        started = datetime.fromisoformat(
            str(started_at_utc).replace("Z", "+00:00")
        )
        completed = datetime.fromisoformat(
            str(completed_at_utc).replace("Z", "+00:00")
        )
    except ValueError:
        return None
    return round((completed - started).total_seconds() * 1000, 3)


def _as_text(value: object) -> str:
    """Return a stable CSV representation for an optional value."""

    if value is None:
        return ""
    return str(value)


def _event_index(
    events: Iterable[dict[str, object]],
    event_type: str,
    key: str,
) -> dict[str, dict[str, object]]:
    """Index the last event of one type by a useful correlation key."""

    result: dict[str, dict[str, object]] = {}
    for event in events:
        if event.get("event_type") != event_type:
            continue
        value = event.get(key)
        if value:
            result[str(value)] = event
    return result


def _prediction_matches(
    prediction: dict[str, str],
    *,
    scenario: str | None,
    execution_id: str | None,
) -> bool:
    """Return whether one master prediction matches report filters."""

    if scenario and prediction.get("scenario", "") != scenario:
        return False
    if (
        execution_id
        and prediction.get("scenario_execution_id", "")
        != execution_id
    ):
        return False
    return True


def _filter_events(
    events: list[dict[str, object]],
    *,
    scenario: str | None,
    execution_id: str | None,
    allowed_transactions: set[str],
) -> list[dict[str, object]]:
    """Filter timing events, including legacy events without execution IDs.

    Older runtime events do not carry ``scenario_execution_id``.  For those
    events an execution filter is correlated through the transaction/capture
    ID found in the master prediction CSV.
    """

    filtered: list[dict[str, object]] = []
    for event in events:
        if scenario and _as_text(event.get("scenario")) != scenario:
            continue
        if not execution_id:
            filtered.append(event)
            continue

        event_execution_id = event.get("scenario_execution_id")
        if event_execution_id:
            if _as_text(event_execution_id) == execution_id:
                filtered.append(event)
            continue

        transaction_id = event.get("transaction_id")
        if transaction_id and _as_text(transaction_id) in allowed_transactions:
            filtered.append(event)
            continue

        capture_id = event.get("capture_id")
        if capture_id and _as_text(capture_id) in allowed_transactions:
            filtered.append(event)
            continue

        pcap_filename = _as_text(event.get("pcap_filename"))
        if pcap_filename:
            capture_stem = Path(pcap_filename).stem
            if capture_stem in allowed_transactions:
                filtered.append(event)

    return filtered


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: Iterable[dict[str, object]],
) -> int:
    """Atomically replace one report CSV and return its row count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(fields),
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: _as_text(row.get(field, ""))
                        for field in fields
                    }
                )
                count += 1
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return count


def _capture_key(event: dict[str, object]) -> str:
    """Choose a stable key for PCAP-related events."""

    return _as_text(
        event.get("capture_id")
        or event.get("pcap_filename")
    )


def _build_pcap_rows(
    events: list[dict[str, object]],
    execution_by_capture: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Merge capture, receiver, and conversion events by PCAP."""

    merged: dict[str, dict[str, object]] = {}
    for event in events:
        if event.get("event_type") not in {
            "pcap_capture",
            "pcap_received",
            "pcap_conversion",
        }:
            continue
        key = _capture_key(event)
        if not key:
            continue
        row = merged.setdefault(key, {"capture_id": key})
        row.update(
            {
                field: value
                for field, value in event.items()
                if value is not None
            }
        )

    rows: list[dict[str, object]] = []
    for key, row in merged.items():
        capture_completed = (
            row.get("capture_completed_at_utc")
            or row.get("capture_finished_at_utc")
        )
        received_at = row.get("received_at_utc")
        conversion_completed = row.get("conversion_completed_at_utc")
        rows.append(
            {
                **row,
                "capture_id": row.get("capture_id") or key,
                "scenario_execution_id": (
                    row.get("scenario_execution_id")
                    or (execution_by_capture or {}).get(key, "")
                ),
                "capture_duration_ms": (
                    row.get("capture_duration_ms")
                    or _timestamp_delta_ms(
                        row.get("capture_started_at_utc"),
                        capture_completed,
                    )
                ),
                "transfer_latency_ms": _timestamp_delta_ms(
                    row.get("upload_started_at_utc")
                    or capture_completed,
                    received_at,
                ),
                "conversion_duration_ms": (
                    row.get("conversion_duration_ms")
                    or _timestamp_delta_ms(
                        row.get("conversion_started_at_utc"),
                        conversion_completed,
                    )
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            _as_text(row.get("capture_started_at_utc")),
            _as_text(row.get("pcap_filename")),
        )
    )
    return rows


def _build_transaction_rows(
    events: list[dict[str, object]],
    execution_by_transaction: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Merge queue, inference, and alert timing events by transaction."""

    merged: dict[str, dict[str, object]] = {}
    for event in events:
        event_type = event.get("event_type")
        if event_type not in {
            "inference_queued",
            "inference_timing",
            "alert_timing",
        }:
            continue
        transaction_id = event.get("transaction_id")
        if not transaction_id:
            continue
        row = merged.setdefault(str(transaction_id), {})
        row.update(
            {
                field: value
                for field, value in event.items()
                if value is not None
            }
        )

    rows: list[dict[str, object]] = []
    for transaction_id, row in merged.items():
        rows.append(
            {
                **row,
                "transaction_id": (
                    row.get("transaction_id") or transaction_id
                ),
                "scenario_execution_id": (
                    row.get("scenario_execution_id")
                    or (execution_by_transaction or {}).get(
                        transaction_id,
                        "",
                    )
                ),
                "queue_wait_duration_ms": (
                    row.get("queue_wait_duration_ms")
                    or _timestamp_delta_ms(
                        row.get("queued_at_utc"),
                        row.get("inference_started_at_utc"),
                    )
                ),
                "inference_to_alert_latency_ms": (
                    row.get("inference_to_alert_latency_ms")
                    or _timestamp_delta_ms(
                        row.get("inference_completed_at_utc"),
                        row.get("alert_recorded_at_utc"),
                    )
                ),
                "end_to_end_detection_latency_ms": (
                    row.get("end_to_end_detection_latency_ms")
                    or _timestamp_delta_ms(
                        row.get("capture_started_at_utc"),
                        row.get("alert_recorded_at_utc"),
                    )
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            _as_text(row.get("queue_sequence")),
            _as_text(row.get("transaction_id")),
        )
    )
    return rows


def _build_window_rows(
    prediction_rows: list[dict[str, str]],
    transaction_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Attach transaction timing to every row in the master prediction CSV."""

    by_transaction = {
        _as_text(row.get("transaction_id")): row
        for row in transaction_rows
        if row.get("transaction_id")
    }

    rows: list[dict[str, object]] = []
    for prediction in prediction_rows:
        label = prediction.get("predicted_label_name", "")
        transaction = by_transaction.get(
            prediction.get("transaction_id", ""),
            {},
        )
        rows.append(
            {
                **prediction,
                "verdict": (
                    prediction.get("verdict")
                    or (
                        "BENIGN"
                        if label == "Benign_Final"
                        else "ATTACK"
                    )
                ),
                "scenario_execution_id": (
                    prediction.get("scenario_execution_id")
                    or transaction.get("scenario_execution_id", "")
                ),
                "queue_wait_duration_ms": transaction.get(
                    "queue_wait_duration_ms", ""
                ),
                "total_inference_duration_ms": transaction.get(
                    "total_inference_duration_ms", ""
                ),
                "pure_model_inference_duration_ms": transaction.get(
                    "pure_model_inference_duration_ms", ""
                ),
                "inference_to_alert_latency_ms": transaction.get(
                    "inference_to_alert_latency_ms", ""
                ),
                "end_to_end_detection_latency_ms": transaction.get(
                    "end_to_end_detection_latency_ms", ""
                ),
            }
        )
    return rows


def generate_timing_reports(
    base_directory: Path,
    *,
    scenario: str | None = None,
    execution_id: str | None = None,
) -> dict[str, Path]:
    """Generate all timing CSVs, optionally filtered by run identity."""

    base_directory = Path(base_directory)
    events = _read_jsonl(
        base_directory / "logs" / "ids_timing_events.jsonl"
    )
    all_prediction_rows = _read_csv(
        base_directory / "inference" / "ids_predictions.csv"
    )
    prediction_rows = [
        prediction
        for prediction in all_prediction_rows
        if _prediction_matches(
            prediction,
            scenario=scenario,
            execution_id=execution_id,
        )
    ]
    execution_by_transaction = {
        _as_text(prediction.get("transaction_id")): _as_text(
            prediction.get("scenario_execution_id")
        )
        for prediction in prediction_rows
        if prediction.get("transaction_id")
        and prediction.get("scenario_execution_id")
    }
    allowed_transactions = set(execution_by_transaction)
    filtered_events = _filter_events(
        events,
        scenario=scenario,
        execution_id=execution_id,
        allowed_transactions=allowed_transactions,
    )
    execution_by_capture = dict(execution_by_transaction)
    pcap_rows = _build_pcap_rows(
        filtered_events,
        execution_by_capture=execution_by_capture,
    )
    transaction_rows = _build_transaction_rows(
        filtered_events,
        execution_by_transaction=execution_by_transaction,
    )
    window_rows = _build_window_rows(
        prediction_rows,
        transaction_rows,
    )

    output_directory = base_directory / "inference"
    outputs = {
        "pcap": output_directory / "ids_pcap_timing.csv",
        "transaction": output_directory / "ids_transaction_timing.csv",
        "window": output_directory / "ids_window_timing.csv",
    }
    counts = {
        "pcap": _write_csv(
            outputs["pcap"],
            PCAP_REPORT_FIELDS,
            pcap_rows,
        ),
        "transaction": _write_csv(
            outputs["transaction"],
            TRANSACTION_REPORT_FIELDS,
            transaction_rows,
        ),
        "window": _write_csv(
            outputs["window"],
            WINDOW_REPORT_FIELDS,
            window_rows,
        ),
    }

    print(
        "[ids-timing-report] Timing events: "
        f"{len(filtered_events)} of {len(events)}; "
        f"predictions: {len(prediction_rows)} of "
        f"{len(all_prediction_rows)}"
    )
    if scenario or execution_id:
        print(
            "[ids-timing-report] Filters: "
            f"scenario={scenario or '*'}, "
            f"execution_id={execution_id or '*'}"
        )
    for name, path in outputs.items():
        print(
            f"[ids-timing-report] {name}: {counts[name]} rows -> {path}"
        )
    return outputs


def main() -> None:
    """Generate reports using the deployed IDS runtime directory."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate read-only IDS timing CSV reports."
    )
    parser.add_argument(
        "--scenario",
        help="Include only one scenario, for example dos_http_flood.",
    )
    parser.add_argument(
        "--execution-id",
        help="Include only one scenario execution ID.",
    )
    args = parser.parse_args()

    base_directory = Path(
        os.environ.get(
            "IDS_BASE_DIR",
            "/home/ubuntu/aws_ids_testbed",
        )
    )
    generate_timing_reports(
        base_directory,
        scenario=args.scenario,
        execution_id=args.execution_id,
    )


if __name__ == "__main__":
    main()
