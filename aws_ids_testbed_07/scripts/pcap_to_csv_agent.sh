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

mkdir -p "$INPUT_DIR" "$CSV_OUTPUT_DIR" "$WORK_DIR" "$CONVERTED_DIR" "$FAILED_DIR"

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

        if "$PYTHON_BIN" "$CONVERTER_DIR/Generating_dataset.py" \
            --pcap-file "$PCAP_FILE" \
            --csv-output-directory "$CSV_OUTPUT_DIR" \
            --work-directory "$WORK_DIR"; then
            date -u +"%Y-%m-%dT%H:%M:%SZ" > "$CONVERTED_MARKER"
            echo "[pcap-to-csv-agent] Converted: $PCAP_FILE"
        else
            date -u +"%Y-%m-%dT%H:%M:%SZ" > "$FAILED_MARKER"
            echo "[pcap-to-csv-agent] Failed: $PCAP_FILE"
        fi
    done
    shopt -u nullglob

    sleep "$SLEEP_SECONDS"
done
