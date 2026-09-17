"""Verify benign connectivity from all five generators to the victim."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_08.benign_generator_lifecycle import (
    BENIGN_GENERATOR_ROLES,
)
from aws_ids_testbed_08.remote_runner import RemoteRunner
from aws_ids_testbed_08.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def verify_benign_connectivity(project_root: Path) -> int:
    """Run small benign connectivity checks from every generator."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    overall_status = 0

    command = r"""
set -u

CONFIG_FILE="/opt/aws_ids_testbed/benign/config/generator.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing configuration: $CONFIG_FILE"
    exit 1
fi

source "$CONFIG_FILE"

export VICTIM_PRIVATE_IP
export VICTIM_UDP_PORT

failed=0

echo "[connectivity] Generator: $GENERATOR_ROLE"
echo "[connectivity] Victim: $VICTIM_PRIVATE_IP"

echo "[connectivity] HTTP..."
if curl --fail --silent --show-error --max-time 5 \
    "http://${VICTIM_PRIVATE_IP}:${VICTIM_HTTP_PORT}/" >/dev/null
then
    echo "HTTP: PASS"
else
    echo "HTTP: FAIL"
    failed=1
fi

echo "[connectivity] HTTPS..."
if curl --insecure --fail --silent --show-error --max-time 5 \
    "https://${VICTIM_PRIVATE_IP}:${VICTIM_HTTPS_PORT}/" >/dev/null
then
    echo "HTTPS: PASS"
else
    echo "HTTPS: FAIL"
    failed=1
fi

echo "[connectivity] DNS..."
if dig +time=2 +tries=1 \
    "@${VICTIM_PRIVATE_IP}" \
    -p "${VICTIM_DNS_PORT}" \
    "${GENERATOR_ROLE}.iot.local" +short | grep -q .
then
    echo "DNS: PASS"
else
    echo "DNS: FAIL"
    failed=1
fi

echo "[connectivity] MQTT..."
if mosquitto_pub \
    -h "${VICTIM_PRIVATE_IP}" \
    -p "${VICTIM_MQTT_PORT}" \
    -t "iot/${GENERATOR_ROLE}/connectivity" \
    -m "connectivity-test"
then
    echo "MQTT: PASS"
else
    echo "MQTT: FAIL"
    failed=1
fi

echo "[connectivity] UDP..."
if python3 - <<'PY'
import os
import socket

host = os.environ["VICTIM_PRIVATE_IP"]
port = int(os.environ["VICTIM_UDP_PORT"])

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.settimeout(3)
client.sendto(b"connectivity-test", (host, port))
response, _ = client.recvfrom(1024)

if response != b"telemetry-ok\n":
    raise SystemExit(1)
PY
then
    echo "UDP: PASS"
else
    echo "UDP: FAIL"
    failed=1
fi

echo "[connectivity] ICMP..."
if ping -c 2 -W 2 "${VICTIM_PRIVATE_IP}" >/dev/null
then
    echo "ICMP: PASS"
else
    echo "ICMP: FAIL"
    failed=1
fi

if [[ "$failed" -ne 0 ]]; then
    echo "[connectivity] One or more checks failed."
    exit 1
fi

echo "[connectivity] All benign connectivity checks passed."
"""

    for role in BENIGN_GENERATOR_ROLES:
        print(f"[benign-connectivity] Testing {role}...")
        status = runner.run_command(
            host=get_public_host(project_root, role),
            command=command,
        )

        if status != 0:
            overall_status = status

    return overall_status
