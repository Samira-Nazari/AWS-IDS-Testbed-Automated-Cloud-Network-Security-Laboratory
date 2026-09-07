#!/usr/bin/env bash
set -euo pipefail

# This script runs on the victim EC2 instance.
# It continuously captures network traffic into small PCAP chunks.
# Each chunk is named using ACTIVE_SCENARIO from victim.env.

CONFIG_FILE="/opt/aws_ids_testbed/config/victim.env"
BASE_DIR="/opt/aws_ids_testbed/pcap"
PENDING_DIR="$BASE_DIR/pending"
WRITING_DIR="$BASE_DIR/writing"
LOG_DIR="/opt/aws_ids_testbed/logs"

mkdir -p "$PENDING_DIR" "$WRITING_DIR" "$LOG_DIR"

echo "[capture-agent] Starting continuous capture agent..."

while true; do
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "[capture-agent] Missing config file: $CONFIG_FILE"
        sleep 5
        continue
    fi

    # Load current victim settings each loop.
    # This lets the controller change ACTIVE_SCENARIO without restarting the agent.
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"

    CAPTURE_ENABLED="${CAPTURE_ENABLED:-false}"
    CAPTURE_INTERFACE="${CAPTURE_INTERFACE:-ens5}"
    CAPTURE_ROTATE_SECONDS="${CAPTURE_ROTATE_SECONDS:-10}"
    CAPTURE_TIMEZONE="${CAPTURE_TIMEZONE:-America/Toronto}"
    ACTIVE_SCENARIO="${ACTIVE_SCENARIO:-unknown}"
    MIN_PCAP_BYTES="${MIN_PCAP_BYTES:-100}"

    if [[ "$CAPTURE_ENABLED" != "true" ]]; then
        echo "[capture-agent] Capture disabled. Waiting..."
        sleep 5
        continue
    fi

    TIMESTAMP="$(TZ="$CAPTURE_TIMEZONE" date +%Y%m%d_%H%M%S)"
    FILE_NAME="${ACTIVE_SCENARIO}_${TIMESTAMP}.pcap"
    TEMP_PATH="/tmp/${FILE_NAME}.tmp"
    FINAL_PATH="$PENDING_DIR/$FILE_NAME"

    echo "[capture-agent] Capturing scenario=$ACTIVE_SCENARIO seconds=$CAPTURE_ROTATE_SECONDS"
    echo "[capture-agent] Writing temporary file: $TEMP_PATH"

    # timeout returns 124 when it stops tcpdump after the requested time.
    # For this script, 124 means the capture chunk completed normally.
    set +e
    sudo timeout "$CAPTURE_ROTATE_SECONDS" tcpdump \
        -i "$CAPTURE_INTERFACE" \
        -w "$TEMP_PATH" \
        tcp port 80
    TCPDUMP_STATUS=$?
    set -e

    if [[ "$TCPDUMP_STATUS" -ne 0 && "$TCPDUMP_STATUS" -ne 124 ]]; then
        echo "[capture-agent] tcpdump failed with exit code: $TCPDUMP_STATUS"
        sudo rm -f "$TEMP_PATH"
        sleep 2
        continue
    fi

    if [[ -f "$TEMP_PATH" ]]; then
        PCAP_SIZE="$(sudo stat -c%s "$TEMP_PATH")"

        if [[ "$PCAP_SIZE" -le "$MIN_PCAP_BYTES" ]]; then
            echo "[capture-agent] Skipping empty PCAP chunk: $TEMP_PATH"
            echo "[capture-agent] Size: $PCAP_SIZE bytes, minimum: $MIN_PCAP_BYTES bytes"
            sudo rm -f "$TEMP_PATH"
            continue
        fi

        sudo mv "$TEMP_PATH" "$FINAL_PATH"
        sudo chown ubuntu:ubuntu "$FINAL_PATH"
        echo "[capture-agent] Completed PCAP chunk: $FINAL_PATH"
    else
        echo "[capture-agent] No PCAP file was created for this chunk."
    fi
done
