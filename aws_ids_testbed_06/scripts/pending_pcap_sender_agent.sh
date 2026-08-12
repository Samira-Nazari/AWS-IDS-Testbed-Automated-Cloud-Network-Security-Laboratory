#!/usr/bin/env bash
set -euo pipefail

# This script runs on the victim EC2 instance.
# It continuously sends completed pending PCAP files to the IDS receiver.
# It works in parallel with continuous_capture_agent.sh.

PENDING_DIR="/opt/aws_ids_testbed/pcap/pending"
SENT_DIR="/opt/aws_ids_testbed/pcap/sent"
LOG_DIR="/opt/aws_ids_testbed/logs"
SENDER_SCRIPT="/opt/aws_ids_testbed/bin/send_pcap_to_ids.sh"
SLEEP_SECONDS=3

mkdir -p "$PENDING_DIR" "$SENT_DIR" "$LOG_DIR"

echo "[sender-agent] Starting pending PCAP sender agent..."

while true; do
    if [[ ! -x "$SENDER_SCRIPT" ]]; then
        echo "[sender-agent] Missing executable sender script: $SENDER_SCRIPT"
        sleep "$SLEEP_SECONDS"
        continue
    fi

    shopt -s nullglob
    PCAP_FILES=("$PENDING_DIR"/*.pcap)
    shopt -u nullglob

    if [[ ${#PCAP_FILES[@]} -eq 0 ]]; then
        sleep "$SLEEP_SECONDS"
        continue
    fi

    for PCAP_PATH in "${PCAP_FILES[@]}"; do
        FILE_NAME="$(basename "$PCAP_PATH")"
        NAME_NO_EXT="${FILE_NAME%.pcap}"

        # Filename format:
        #   scenario_YYYYMMDD_HHMMSS.pcap
        #
        # Remove the timestamp to recover the scenario label.
        SCENARIO="$(echo "$NAME_NO_EXT" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')"

        if [[ -z "$SCENARIO" || "$SCENARIO" == "$NAME_NO_EXT" ]]; then
            SCENARIO="unknown"
        fi

        echo "[sender-agent] Sending pending PCAP: $PCAP_PATH"
        echo "[sender-agent] Scenario: $SCENARIO"

        if "$SENDER_SCRIPT" "$PCAP_PATH" "$SCENARIO"; then
            mv "$PCAP_PATH" "$SENT_DIR/$FILE_NAME"
            echo "[sender-agent] Moved to sent: $SENT_DIR/$FILE_NAME"
        else
            echo "[sender-agent] Upload failed. Keeping in pending: $PCAP_PATH"
        fi
    done

    sleep "$SLEEP_SECONDS"
done
