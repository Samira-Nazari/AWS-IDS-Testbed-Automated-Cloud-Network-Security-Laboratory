#!/usr/bin/env bash

# send_pcap_to_ids.sh
# This script runs on the victim EC2 instance.
# It sends one completed PCAP file from victim to the IDS FastAPI receiver.

set -euo pipefail

CONFIG_FILE="/opt/aws_ids_testbed/config/victim.env"

utc_now_iso() {
    date -u +"%Y-%m-%dT%H:%M:%S.%NZ"
}

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 PCAP_PATH [SCENARIO]"
    exit 1
fi

PCAP_PATH="$1"
SCENARIO="${2:-unknown}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[victim-sender] Missing config file: $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

if [[ -z "${IDS_RECEIVER_URL:-}" ]]; then
    echo "[victim-sender] IDS_RECEIVER_URL is missing in $CONFIG_FILE"
    exit 1
fi

if [[ ! -f "$PCAP_PATH" ]]; then
    echo "[victim-sender] PCAP file not found: $PCAP_PATH"
    exit 1
fi

TIMING_METADATA_PATH="${PCAP_PATH}.timing.json"

if [[ -f "$TIMING_METADATA_PATH" ]]; then
    TIMING_METADATA="$(<"$TIMING_METADATA_PATH")"
else
    echo "[victim-sender] Timing metadata not found: $TIMING_METADATA_PATH"
    TIMING_METADATA="{}"
fi

UPLOAD_STARTED_AT_UTC="$(utc_now_iso)"

echo "[victim-sender] Sending PCAP to IDS..."
echo "[victim-sender] PCAP: $PCAP_PATH"
echo "[victim-sender] Scenario: $SCENARIO"
echo "[victim-sender] IDS: $IDS_RECEIVER_URL"

curl --fail --silent --show-error \
    --max-time 30 \
    -F "file=@${PCAP_PATH}" \
    -F "source_role=victim" \
    -F "scenario=${SCENARIO}" \
    --form-string "timing_metadata=${TIMING_METADATA}" \
    --form-string "upload_started_at_utc=${UPLOAD_STARTED_AT_UTC}" \
    "${IDS_RECEIVER_URL}/upload-pcap"

UPLOAD_COMPLETED_AT_UTC="$(utc_now_iso)"
echo
echo "[victim-sender] Upload completed."
echo "[victim-sender] Upload started: $UPLOAD_STARTED_AT_UTC"
echo "[victim-sender] Upload completed: $UPLOAD_COMPLETED_AT_UTC"
