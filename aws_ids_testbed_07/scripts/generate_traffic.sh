#!/usr/bin/env bash

# generate_traffic.sh
# This script runs on the attacker EC2 instance.
# It generates selected lab traffic toward the victim EC2 instance.

set -euo pipefail

CONFIG_FILE="/opt/aws_ids_testbed/config/attacker.env"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 TRAFFIC_CODE [--requests N] [--concurrency N] [--packet-count N] [--port N]"
    echo "Traffic codes:"
    echo "  1 = benign_http"
    echo "  2 = dos_http_flood"
    echo "  3 = dos_syn_flood"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[attacker-traffic] Missing config file: $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

TRAFFIC_CODE="$1"
shift

REQUESTS=""
CONCURRENCY=""
PACKET_COUNT=""
PORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --requests)
            REQUESTS="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --packet-count)
            PACKET_COUNT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "[attacker-traffic] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${VICTIM_PRIVATE_IP:-}" ]]; then
    echo "[attacker-traffic] VICTIM_PRIVATE_IP is missing in $CONFIG_FILE"
    exit 1
fi

if [[ -z "${VICTIM_URL:-}" ]]; then
    echo "[attacker-traffic] VICTIM_URL is missing in $CONFIG_FILE"
    exit 1
fi

case "$TRAFFIC_CODE" in
    1)
        TRAFFIC_NAME="benign_http"
        REQUESTS="${REQUESTS:-${DEFAULT_HTTP_REQUESTS:-100}}"
        CONCURRENCY="${CONCURRENCY:-${DEFAULT_HTTP_CONCURRENCY:-5}}"

        echo "[attacker-traffic] Traffic type: $TRAFFIC_NAME"
        echo "[attacker-traffic] Requests: $REQUESTS"
        echo "[attacker-traffic] Concurrency: $CONCURRENCY"
        echo "[attacker-traffic] Target: $VICTIM_URL/"

        ab -n "$REQUESTS" -c "$CONCURRENCY" "$VICTIM_URL/"
        ;;

    2)
        TRAFFIC_NAME="dos_http_flood"
        REQUESTS="${REQUESTS:-${DEFAULT_DOS_HTTP_REQUESTS:-1000}}"
        CONCURRENCY="${CONCURRENCY:-${DEFAULT_DOS_HTTP_CONCURRENCY:-50}}"

        echo "[attacker-traffic] Traffic type: $TRAFFIC_NAME"
        echo "[attacker-traffic] Requests: $REQUESTS"
        echo "[attacker-traffic] Concurrency: $CONCURRENCY"
        echo "[attacker-traffic] Target: $VICTIM_URL/"

        ab -n "$REQUESTS" -c "$CONCURRENCY" "$VICTIM_URL/"
        ;;

    3)
        TRAFFIC_NAME="dos_syn_flood"
        PACKET_COUNT="${PACKET_COUNT:-${DEFAULT_SYN_PACKET_COUNT:-100}}"
        PORT="${PORT:-${DEFAULT_TARGET_PORT:-80}}"

        echo "[attacker-traffic] Traffic type: $TRAFFIC_NAME"
        echo "[attacker-traffic] Packet count: $PACKET_COUNT"
        echo "[attacker-traffic] Port: $PORT"
        echo "[attacker-traffic] Target IP: $VICTIM_PRIVATE_IP"

        sudo hping3 -S -p "$PORT" -c "$PACKET_COUNT" "$VICTIM_PRIVATE_IP"
        ;;

    *)
        echo "[attacker-traffic] Invalid traffic code: $TRAFFIC_CODE"
        echo "Use:"
        echo "  1 = benign_http"
        echo "  2 = dos_http_flood"
        echo "  3 = dos_syn_flood"
        exit 1
        ;;
esac

echo "[attacker-traffic] Traffic generation completed."
