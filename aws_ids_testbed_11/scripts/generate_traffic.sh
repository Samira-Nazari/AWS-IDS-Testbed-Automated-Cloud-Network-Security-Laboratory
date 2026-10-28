#!/usr/bin/env bash

# generate_traffic.sh
# This script runs on the attacker EC2 instance.
# It generates selected lab traffic toward the victim EC2 instance.

set -euo pipefail

CONFIG_FILE="/opt/aws_ids_testbed/config/attacker.env"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 TRAFFIC_CODE [--requests N] [--concurrency N] [--packet-count N] [--port N] [--interval-microseconds N]"
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
INTERVAL_MICROSECONDS=""

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
        --interval-microseconds)
            INTERVAL_MICROSECONDS="$2"
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
        CONCURRENCY="${CONCURRENCY:-${DEFAULT_HTTP_CONCURRENCY:-1}}"

        if [[ "$CONCURRENCY" != "1" ]]; then
            echo "[attacker-traffic] Benign traffic forces concurrency to 1."
        fi
        CONCURRENCY=1

        BENIGN_PATHS=(
            "/"
            "/index.html"
            "/favicon.ico"
            "/?view=home"
            "/index.html?source=browser"
        )

        BENIGN_DELAYS=(
            "0.80"
            "1.10"
            "1.30"
            "1.50"
            "1.80"
        )

        USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"

        echo "[attacker-traffic] Traffic type: $TRAFFIC_NAME"
        echo "[attacker-traffic] Requests: $REQUESTS"
        echo "[attacker-traffic] Effective concurrency: $CONCURRENCY"
        echo "[attacker-traffic] Target: $VICTIM_URL"
        echo "[attacker-traffic] Sending sequential browser-like requests..."

        for ((REQUEST_NUMBER = 1; REQUEST_NUMBER <= REQUESTS; REQUEST_NUMBER++)); do
            PATH_INDEX=$(((REQUEST_NUMBER - 1) % ${#BENIGN_PATHS[@]}))
            DELAY_INDEX=$(((REQUEST_NUMBER - 1) % ${#BENIGN_DELAYS[@]}))

            REQUEST_PATH="${BENIGN_PATHS[$PATH_INDEX]}"
            REQUEST_DELAY="${BENIGN_DELAYS[$DELAY_INDEX]}"

            curl \
                --silent \
                --show-error \
                --location \
                --output /dev/null \
                --connect-timeout 5 \
                --max-time 10 \
                --user-agent "$USER_AGENT" \
                --header "Accept: text/html,application/xhtml+xml,image/avif,image/webp,*/*;q=0.8" \
                --header "Accept-Language: en-US,en;q=0.9" \
                "${VICTIM_URL}${REQUEST_PATH}"

            if ((REQUEST_NUMBER == 1 ||
                 REQUEST_NUMBER % 10 == 0 ||
                 REQUEST_NUMBER == REQUESTS)); then
                echo "[attacker-traffic] Completed request $REQUEST_NUMBER/$REQUESTS"
            fi

            if ((REQUEST_NUMBER < REQUESTS)); then
                sleep "$REQUEST_DELAY"
            fi
        done
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
        PACKET_COUNT="${PACKET_COUNT:-${DEFAULT_SYN_PACKET_COUNT:-1000}}"
        PORT="${PORT:-${DEFAULT_TARGET_PORT:-80}}"
        INTERVAL_MICROSECONDS="${INTERVAL_MICROSECONDS:-${DEFAULT_SYN_INTERVAL_MICROSECONDS:-5000}}"

        if ! [[ "$PACKET_COUNT" =~ ^[1-9][0-9]*$ ]]; then
            echo "[attacker-traffic] Packet count must be a positive integer."
            exit 1
        fi

        if ! [[ "$INTERVAL_MICROSECONDS" =~ ^[1-9][0-9]*$ ]]; then
            echo "[attacker-traffic] SYN interval must be a positive integer."
            exit 1
        fi

        echo "[attacker-traffic] Traffic type: $TRAFFIC_NAME"
        echo "[attacker-traffic] Packet count: $PACKET_COUNT"
        echo "[attacker-traffic] Port: $PORT"
        echo "[attacker-traffic] Interval: ${INTERVAL_MICROSECONDS} microseconds"
        echo "[attacker-traffic] Target IP: $VICTIM_PRIVATE_IP"

        sudo hping3 -q -S -p "$PORT" -c "$PACKET_COUNT" \
            -i "u${INTERVAL_MICROSECONDS}" "$VICTIM_PRIVATE_IP"
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
