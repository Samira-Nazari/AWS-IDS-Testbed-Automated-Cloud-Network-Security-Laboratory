#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${IDS_BASE_DIR:-$HOME/aws_ids_testbed}"
INPUT_DIR="${PCAP_INPUT_DIR:-$BASE_DIR/input}"
CSV_OUTPUT_DIR="${CSV_OUTPUT_DIR:-$BASE_DIR/output/csv}"
WORK_DIR="${PCAP_TO_CSV_WORK_DIR:-$BASE_DIR/tmp/pcap_to_csv}"
CONVERTER_DIR="${PCAP_TO_CSV_CONVERTER_DIR:-$BASE_DIR/ids_pcap_to_csv/pcap2csv}"
PYTHON_BIN="${IDS_PYTHON_BIN:-$BASE_DIR/ids_env/bin/python}"
STATE_DIR="${PCAP_TO_CSV_STATE_DIR:-$BASE_DIR/state/pcap_to_csv}"
CONVERTED_DIR="$STATE_DIR/converted"
FAILED_DIR="$STATE_DIR/failed"
SLEEP_SECONDS="${PCAP_TO_CSV_SCAN_SECONDS:-2}"
TIMING_UTILITY_PATH="$BASE_DIR/ids_timing.py"
TIMING_EVENTS_PATH="$BASE_DIR/logs/ids_timing_events.jsonl"

mkdir -p "$INPUT_DIR" "$CSV_OUTPUT_DIR" "$WORK_DIR" "$CONVERTED_DIR" "$FAILED_DIR"

utc_now_iso() {
    date -u +"%Y-%m-%dT%H:%M:%S.%NZ"
}

record_conversion_event() {
    local pcap_path="$1"
    local csv_path="$2"
    local status="$3"
    local started_at_utc="$4"
    local completed_at_utc="$5"
    local duration_ms="$6"

    # Timing is observational. A missing timing helper must never stop CSV conversion.
    if [[ ! -f "$TIMING_UTILITY_PATH" ]]; then
        echo "[pcap-to-csv-agent] Timing utility not found: $TIMING_UTILITY_PATH"
        return 0
    fi

    if ! BASE_DIR="$BASE_DIR" \
        TIMING_EVENTS_PATH="$TIMING_EVENTS_PATH" \
        PCAP_PATH="$pcap_path" \
        CSV_PATH="$csv_path" \
        CONVERSION_STATUS="$status" \
        CONVERSION_STARTED_AT_UTC="$started_at_utc" \
        CONVERSION_COMPLETED_AT_UTC="$completed_at_utc" \
        CONVERSION_DURATION_MS="$duration_ms" \
        PYTHONPATH="$BASE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from ids_timing import append_timing_event

base_dir = Path(os.environ["BASE_DIR"])
pcap_path = Path(os.environ["PCAP_PATH"])
csv_path = Path(os.environ["CSV_PATH"])
metadata_path = pcap_path.with_name(f"{pcap_path.name}.timing.json")

metadata = {}
if metadata_path.is_file():
    try:
        candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            metadata = candidate
    except (OSError, json.JSONDecodeError):
        metadata = {"timing_metadata_parse_error": True}

event = dict(metadata)
event.update(
    {
        "event_type": "pcap_conversion",
        "pcap_filename": pcap_path.name,
        "csv_filename": csv_path.name,
        "csv_path": str(csv_path),
        "conversion_status": os.environ["CONVERSION_STATUS"],
        "conversion_started_at_utc": os.environ["CONVERSION_STARTED_AT_UTC"],
        "conversion_completed_at_utc": os.environ["CONVERSION_COMPLETED_AT_UTC"],
        "conversion_duration_ms": float(os.environ["CONVERSION_DURATION_MS"]),
        "csv_exists": csv_path.is_file(),
    }
)

if csv_path.is_file():
    event["csv_size_bytes"] = csv_path.stat().st_size

append_timing_event(Path(os.environ["TIMING_EVENTS_PATH"]), event)
PY
    then
        echo "[pcap-to-csv-agent] Timing event recorded: $pcap_path ($status)"
    else
        echo "[pcap-to-csv-agent] Could not record timing event: $pcap_path"
    fi
}

echo "[pcap-to-csv-agent] Starting PCAP to CSV agent..."
echo "[pcap-to-csv-agent] Input: $INPUT_DIR"
echo "[pcap-to-csv-agent] CSV output: $CSV_OUTPUT_DIR"
echo "[pcap-to-csv-agent] Converter: $CONVERTER_DIR"

while true; do
    shopt -s nullglob
    for PCAP_FILE in "$INPUT_DIR"/*.pcap; do
        BASENAME="$(basename "$PCAP_FILE")"
        CONVERTED_MARKER="$CONVERTED_DIR/$BASENAME.done"
        FAILED_MARKER="$FAILED_DIR/$BASENAME.failed"

        if [ -f "$CONVERTED_MARKER" ]; then
            continue
        fi

        if [ -f "$FAILED_MARKER" ]; then
            continue
        fi

        echo "[pcap-to-csv-agent] New PCAP detected: $PCAP_FILE"
        CSV_FILE="$CSV_OUTPUT_DIR/${BASENAME%.pcap}.csv"
        CONVERSION_STARTED_AT_UTC="$(utc_now_iso)"
        CONVERSION_STARTED_EPOCH_MS="$(date +%s%3N)"

        if "$PYTHON_BIN" "$CONVERTER_DIR/Generating_dataset.py" \
            --pcap-file "$PCAP_FILE" \
            --csv-output-directory "$CSV_OUTPUT_DIR" \
            --work-directory "$WORK_DIR"; then
            CONVERSION_COMPLETED_AT_UTC="$(utc_now_iso)"
            CONVERSION_COMPLETED_EPOCH_MS="$(date +%s%3N)"
            CONVERSION_DURATION_MS=$((CONVERSION_COMPLETED_EPOCH_MS - CONVERSION_STARTED_EPOCH_MS))
            date -u +"%Y-%m-%dT%H:%M:%SZ" > "$CONVERTED_MARKER"
            record_conversion_event \
                "$PCAP_FILE" \
                "$CSV_FILE" \
                "success" \
                "$CONVERSION_STARTED_AT_UTC" \
                "$CONVERSION_COMPLETED_AT_UTC" \
                "$CONVERSION_DURATION_MS"
            echo "[pcap-to-csv-agent] Converted: $PCAP_FILE"
        else
            CONVERSION_COMPLETED_AT_UTC="$(utc_now_iso)"
            CONVERSION_COMPLETED_EPOCH_MS="$(date +%s%3N)"
            CONVERSION_DURATION_MS=$((CONVERSION_COMPLETED_EPOCH_MS - CONVERSION_STARTED_EPOCH_MS))
            date -u +"%Y-%m-%dT%H:%M:%SZ" > "$FAILED_MARKER"
            record_conversion_event \
                "$PCAP_FILE" \
                "$CSV_FILE" \
                "failed" \
                "$CONVERSION_STARTED_AT_UTC" \
                "$CONVERSION_COMPLETED_AT_UTC" \
                "$CONVERSION_DURATION_MS"
            echo "[pcap-to-csv-agent] Failed: $PCAP_FILE"
        fi
    done
    shopt -u nullglob

    sleep "$SLEEP_SECONDS"
done
