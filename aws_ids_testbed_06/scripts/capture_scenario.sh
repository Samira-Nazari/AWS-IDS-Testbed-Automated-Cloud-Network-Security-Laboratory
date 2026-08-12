#!/usr/bin/env bash

# capture_scenario.sh
# This script runs on the victim EC2 instance.
# It captures traffic for one scenario and creates a completed PCAP file.

set -euo pipefail

CONFIG_FILE="/opt/aws_ids_testbed/config/victim.env"
BASE_DIR="/opt/aws_ids_testbed"
WRITING_DIR="$BASE_DIR/pcap/writing"
PENDING_DIR="$BASE_DIR/pcap/pending"
LOG_DIR="$BASE_DIR/logs"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 SCENARIO [SECONDS]"
    exit 1
fi

SCENARIO="$1"
REQUESTED_SECONDS="${2:-}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[victim-capture] Missing config file: $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

CAPTURE_INTERFACE="${CAPTURE_INTERFACE:-ens5}"
CAPTURE_SECONDS="${REQUESTED_SECONDS:-${CAPTURE_SECONDS:-20}}"
CAPTURE_TIMEZONE="${CAPTURE_TIMEZONE:-America/Toronto}"

TIMESTAMP="$(TZ="$CAPTURE_TIMEZONE" date +%Y%m%d_%H%M%S)"
# tcpdump may drop privileges while writing.
# Writing the temporary file in /tmp avoids permission problems.
TMP_PATH="/tmp/${SCENARIO}_${TIMESTAMP}.pcap.tmp"
FINAL_PATH="$PENDING_DIR/${SCENARIO}_${TIMESTAMP}.pcap"
LOG_PATH="$LOG_DIR/capture_${SCENARIO}_${TIMESTAMP}.log"

mkdir -p "$WRITING_DIR" "$PENDING_DIR" "$LOG_DIR"

echo "[victim-capture] Scenario: $SCENARIO"
echo "[victim-capture] Interface: $CAPTURE_INTERFACE"
echo "[victim-capture] Seconds: $CAPTURE_SECONDS"
echo "[victim-capture] Temporary PCAP: $TMP_PATH"
echo "[victim-capture] Final PCAP: $FINAL_PATH"

sudo timeout "$CAPTURE_SECONDS" tcpdump \
    -i "$CAPTURE_INTERFACE" \
    -U \
    -w "$TMP_PATH" \
    > "$LOG_PATH" \
    2>&1 || status=$?

status="${status:-0}"

# timeout returns 124 when it stops tcpdump after the requested time.
# For our capture workflow, that is a normal successful result.
if [[ "$status" != "0" && "$status" != "124" ]]; then
    echo "[victim-capture] tcpdump failed with exit code: $status"
    echo "[victim-capture] Log:"
    cat "$LOG_PATH"
    exit "$status"
fi

sudo chown ubuntu:ubuntu "$TMP_PATH"
mv "$TMP_PATH" "$FINAL_PATH"

echo "[victim-capture] Capture completed."
echo "[victim-capture] Pending PCAP: $FINAL_PATH"
