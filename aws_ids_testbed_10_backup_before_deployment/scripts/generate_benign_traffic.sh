#!/usr/bin/env bash

# generate_benign_traffic.sh
# Run one normal benign traffic profile based on GENERATOR_ROLE.

set -euo pipefail

CONFIG_FILE="/opt/aws_ids_testbed/benign/config/generator.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing configuration: $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

: "${GENERATOR_ROLE:?Missing GENERATOR_ROLE}"
: "${VICTIM_PRIVATE_IP:?Missing VICTIM_PRIVATE_IP}"

if [[ $# -ne 2 || "$1" != "--duration-seconds" ]]; then
    echo "Usage: $0 --duration-seconds SECONDS"
    exit 1
fi

DURATION_SECONDS="$2"

if ! [[ "$DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Duration must be a positive integer."
    exit 1
fi

END_TIME=$(( $(date +%s) + DURATION_SECONDS ))

HTTP_BASE="http://${VICTIM_PRIVATE_IP}:${VICTIM_HTTP_PORT}"
HTTPS_BASE="https://${VICTIM_PRIVATE_IP}:${VICTIM_HTTPS_PORT}"

# Slow randomized benign profile.
HTTPS_WORKERS="${BENIGN_HTTPS_WORKERS:-1}"
HTTPS_MIN_SLEEP="${BENIGN_HTTPS_MIN_SLEEP:-20.0}"
HTTPS_MAX_SLEEP="${BENIGN_HTTPS_MAX_SLEEP:-60.0}"
DNS_UDP_MIN_SLEEP="${BENIGN_DNS_UDP_MIN_SLEEP:-30.0}"
DNS_UDP_MAX_SLEEP="${BENIGN_DNS_UDP_MAX_SLEEP:-90.0}"
HTTP_MIN_SLEEP="${BENIGN_HTTP_MIN_SLEEP:-30.0}"
HTTP_MAX_SLEEP="${BENIGN_HTTP_MAX_SLEEP:-90.0}"
MQTT_MIN_SLEEP="${BENIGN_MQTT_MIN_SLEEP:-30.0}"
MQTT_MAX_SLEEP="${BENIGN_MQTT_MAX_SLEEP:-120.0}"
ICMP_UDP_MIN_SLEEP="${BENIGN_ICMP_UDP_MIN_SLEEP:-45.0}"
ICMP_UDP_MAX_SLEEP="${BENIGN_ICMP_UDP_MAX_SLEEP:-180.0}"
STARTUP_JITTER_MAX="${BENIGN_STARTUP_JITTER_MAX:-15.0}"


cleanup() {
    for child_pid in $(jobs -pr 2>/dev/null || true); do
        kill "$child_pid" 2>/dev/null || true
    done
}


trap cleanup EXIT INT TERM


traffic_active() {
    (( $(date +%s) < END_TIME ))
}


sleep_random() {
    python3 - "$1" "$2" "$END_TIME" <<'PY'
import random
import sys
import time

minimum = float(sys.argv[1])
maximum = float(sys.argv[2])
end_time = float(sys.argv[3])

remaining = end_time - time.time()

if remaining > 0:
    time.sleep(min(random.uniform(minimum, maximum), remaining))
PY
}


https_worker() {
    sleep_random 0.0 "$STARTUP_JITTER_MAX"

    while traffic_active; do
        curl \
            --insecure \
            --silent \
            --show-error \
            --max-time 15 \
            --user-agent "IoT-Camera/1.0" \
            "${HTTPS_BASE}/index.html?camera=$RANDOM" \
            >/dev/null 2>&1 || true

        sleep_random "$HTTPS_MIN_SLEEP" "$HTTPS_MAX_SLEEP"
    done
}


run_https_traffic() {
    echo "[benign] Starting HTTPS traffic workers."

    for ((worker = 1; worker <= HTTPS_WORKERS; worker++)); do
        https_worker &
    done

    wait
}


run_dns_udp_traffic() {
    echo "[benign] Starting DNS and UDP telemetry traffic."
    sleep_random 0.0 "$STARTUP_JITTER_MAX"

    while traffic_active; do
        dig \
            +time=2 \
            +tries=1 \
            "@${VICTIM_PRIVATE_IP}" \
            -p "${VICTIM_DNS_PORT}" \
            "${GENERATOR_ROLE}.iot.local" \
            +short \
            >/dev/null 2>&1 || true

        python3 - "${VICTIM_PRIVATE_IP}" "${VICTIM_UDP_PORT}" <<'PY' || true
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.settimeout(3)
client.sendto(b"sensor-telemetry", (host, port))

try:
    client.recvfrom(1024)
except socket.timeout:
    pass
PY

        sleep_random "$DNS_UDP_MIN_SLEEP" "$DNS_UDP_MAX_SLEEP"
    done
}


run_http_traffic() {
    echo "[benign] Starting HTTP traffic."
    sleep_random 0.0 "$STARTUP_JITTER_MAX"

    while traffic_active; do
        curl \
            --silent \
            --show-error \
            --max-time 15 \
            --user-agent "IoT-Browser/1.0" \
            "${HTTP_BASE}/index.html?view=$RANDOM" \
            >/dev/null 2>&1 || true

        sleep_random "$HTTP_MIN_SLEEP" "$HTTP_MAX_SLEEP"
    done
}


run_mqtt_traffic() {
    echo "[benign] Starting MQTT traffic."
    sleep_random 0.0 "$STARTUP_JITTER_MAX"

    mosquitto_sub \
        -h "${VICTIM_PRIVATE_IP}" \
        -p "${VICTIM_MQTT_PORT}" \
        -t "iot/${GENERATOR_ROLE}/commands" \
        -k 60 \
        >/dev/null 2>&1 &

    while traffic_active; do
        mosquitto_pub \
            -h "${VICTIM_PRIVATE_IP}" \
            -p "${VICTIM_MQTT_PORT}" \
            -t "iot/${GENERATOR_ROLE}/telemetry" \
            -m "{\"device\":\"${GENERATOR_ROLE}\",\"value\":$RANDOM}" \
            >/dev/null 2>&1 || true

        sleep_random "$MQTT_MIN_SLEEP" "$MQTT_MAX_SLEEP"
    done
}


run_icmp_udp_traffic() {
    echo "[benign] Starting ICMP and UDP heartbeat traffic."
    sleep_random 0.0 "$STARTUP_JITTER_MAX"

    while traffic_active; do
        ping \
            -c 1 \
            -W 2 \
            "${VICTIM_PRIVATE_IP}" \
            >/dev/null 2>&1 || true

        python3 - "${VICTIM_PRIVATE_IP}" "${VICTIM_UDP_PORT}" <<'PY' || true
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.settimeout(2)
client.sendto(b"heartbeat", (host, port))
PY

        sleep_random "$ICMP_UDP_MIN_SLEEP" "$ICMP_UDP_MAX_SLEEP"
    done
}


echo "[benign] Generator role: ${GENERATOR_ROLE}"

case "${GENERATOR_ROLE}" in
    benign_generator_01)
        run_https_traffic
        ;;

    benign_generator_02)
        run_dns_udp_traffic
        ;;

    benign_generator_03)
        run_http_traffic
        ;;

    benign_generator_04)
        run_mqtt_traffic
        ;;

    benign_generator_05)
        run_icmp_udp_traffic
        ;;

    *)
        echo "Unknown generator role: ${GENERATOR_ROLE}"
        exit 1
        ;;
esac
