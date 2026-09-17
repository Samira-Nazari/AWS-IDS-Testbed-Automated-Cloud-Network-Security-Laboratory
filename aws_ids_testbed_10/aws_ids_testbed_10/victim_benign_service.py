"""Deploy and run victim-side benign service setup."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_10.remote_runner import RemoteRunner
from aws_ids_testbed_10.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def setup_victim_benign_services(project_root: Path) -> int:
    """Upload and run the victim benign-service setup script."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")
    script_path = project_root / "scripts" / "setup_victim_benign_services.sh"

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_script(
        host=victim_public_host,
        local_script_path=script_path,
    )


def verify_victim_benign_services(project_root: Path) -> int:
    """Verify victim benign services and listening ports."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = r"""
set -u

failed=0

CONFIG_FILE="/opt/aws_ids_testbed/config/benign.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing configuration: $CONFIG_FILE"
    exit 1
fi
source "$CONFIG_FILE"

for service in nginx dnsmasq mosquitto aws-ids-benign-udp; do
    status="$(systemctl is-active "$service" 2>/dev/null || true)"
    printf "%-24s %s\n" "$service" "$status"

    if [[ "$status" != "active" ]]; then
        failed=1
    fi
done

echo "--- listening ports ---"
sudo ss -lntup | grep -E ':(53|80|443|1883|9999)[[:space:]]' || failed=1

echo "--- MQTT network binding ---"
if sudo ss -lntH | awk \
    -v port=":${VICTIM_MQTT_PORT}" \
    '$4 ~ port && $4 !~ /127\\.0\\.0\\.1/ && $4 !~ /\\[::1\\]/ {found=1}
     END {exit(found ? 0 : 1)}'
then
    echo "MQTT is reachable on a non-loopback address."
else
    echo "MQTT is still bound only to localhost."
    failed=1
fi

if [[ "$failed" -ne 0 ]]; then
    echo "Victim benign service verification failed."
    exit 1
fi

echo "Victim benign services and ports are ready."
"""

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )
