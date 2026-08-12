#!/usr/bin/env bash

# send_pcap_to_ids.sh
# This script runs on the victim EC2 instance.
# It sends one completed PCAP file from victim to the IDS FastAPI receiver.

set -euo pipefail

CONFIG_FILE="/opt/aws_ids_testbed/config/victim.env"

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

echo "[victim-sender] Sending PCAP to IDS..."
echo "[victim-sender] PCAP: $PCAP_PATH"
echo "[victim-sender] Scenario: $SCENARIO"
echo "[victim-sender] IDS: $IDS_RECEIVER_URL"

curl --fail --silent --show-error \
    --max-time 30 \
    -F "file=@${PCAP_PATH}" \
    -F "source_role=victim" \
    -F "scenario=${SCENARIO}" \
    "${IDS_RECEIVER_URL}/upload-pcap"

echo
echo "[victim-sender] Upload completed."
