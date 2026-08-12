"""Deploy victim-side sender scripts to the victim EC2 instance."""

from __future__ import annotations

import shlex
from pathlib import Path

from aws_ids_testbed_06.remote_runner import RemoteRunner
from aws_ids_testbed_06.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def deploy_victim_sender(project_root: Path) -> int:
    """Copy the victim PCAP sender script to the victim EC2 instance.

    This function runs from your local controller machine.

    It copies:

        scripts/send_pcap_to_ids.sh

    to the victim path:

        /opt/aws_ids_testbed/bin/send_pcap_to_ids.sh
    """
    import paramiko
    from scp import SCPClient

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    local_sender_path = project_root / "scripts" / "send_pcap_to_ids.sh"
    remote_bin_dir = "/opt/aws_ids_testbed/bin"
    remote_sender_path = f"{remote_bin_dir}/send_pcap_to_ids.sh"

    if not private_key_path.exists():
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

    if not local_sender_path.exists():
        raise FileNotFoundError(f"Victim sender script not found: {local_sender_path}")

    ssh_client = paramiko.SSHClient()

    # Use known_hosts when possible.
    ssh_client.load_system_host_keys()

    # In this private lab, automatically accept a new EC2 host key.
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {username}@{victim_public_host} ...")
    ssh_client.connect(
        hostname=victim_public_host,
        username=username,
        key_filename=str(private_key_path),
        timeout=30,
    )

    try:
        print("Creating victim sender folder...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=f"sudo mkdir -p {remote_bin_dir} && sudo chown ubuntu:ubuntu {remote_bin_dir}",
        )

        print("Uploading victim sender script...")
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.put(str(local_sender_path), remote_sender_path)

        print("Making victim sender executable...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=f"chmod 700 {remote_sender_path}",
        )

        print("Victim sender deployed.")
        return 0

    finally:
        ssh_client.close()


def verify_victim_sender(project_root: Path) -> int:
    """Verify that the victim sender script is installed correctly.

    This function does not hard-code the victim IP.

    It checks three things on the victim:

    1. The sender script exists.
    2. The sender script is executable.
    3. The sender script has valid bash syntax.

    It does not run the sender without arguments, because that causes
    a usage error and exit code 1.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    sender_path = "/opt/aws_ids_testbed/bin/send_pcap_to_ids.sh"

    command = (
        f"ls -lh {sender_path} && "
        f"test -f {sender_path} && "
        f"test -x {sender_path} && "
        f"bash -n {sender_path} && "
        "echo 'Victim sender is installed, executable, and has valid bash syntax.'"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def victim_send_pcap(project_root: Path, pcap_path: str, scenario: str) -> int:
    """Send any PCAP file from victim to IDS.

    This function does not hard-code the victim IP.

    It reads the victim public IP from inventory.yaml.
    Then it runs the victim-side sender script.

    The victim-side sender reads IDS_RECEIVER_URL from:

        /opt/aws_ids_testbed/config/victim.env
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = (
        "/opt/aws_ids_testbed/bin/send_pcap_to_ids.sh "
        f"{shlex.quote(pcap_path)} "
        f"{shlex.quote(scenario)}"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def victim_send_pending_pcaps(project_root: Path) -> int:
    """Send all pending PCAP files from victim to IDS.

    This function does not hard-code the victim IP, PCAP filename, or scenario.

    It looks inside:

        /opt/aws_ids_testbed/pcap/pending

    For every .pcap file it finds:
    - infer scenario from filename
    - upload the file to IDS
    - move successfully uploaded files to sent/
    - keep failed uploads in pending/
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = r"""
set -euo pipefail

PENDING_DIR="/opt/aws_ids_testbed/pcap/pending"
SENT_DIR="/opt/aws_ids_testbed/pcap/sent"
FAILED_COUNT=0
SENT_COUNT=0

mkdir -p "$PENDING_DIR" "$SENT_DIR"

shopt -s nullglob
PCAP_FILES=("$PENDING_DIR"/*.pcap)

if [[ ${#PCAP_FILES[@]} -eq 0 ]]; then
    echo "[victim-sender] No pending PCAP files found."
    exit 0
fi

for PCAP_PATH in "${PCAP_FILES[@]}"; do
    FILE_NAME="$(basename "$PCAP_PATH")"
    NAME_NO_EXT="${FILE_NAME%.pcap}"

    # Filename format:
    #   scenario_YYYYMMDD_HHMMSS.pcap
    #
    # Remove the last two underscore parts to recover the scenario.
    SCENARIO="$(echo "$NAME_NO_EXT" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')"

    if [[ -z "$SCENARIO" || "$SCENARIO" == "$NAME_NO_EXT" ]]; then
        SCENARIO="unknown"
    fi

    echo "[victim-sender] Sending pending PCAP: $PCAP_PATH"
    echo "[victim-sender] Scenario: $SCENARIO"

    if /opt/aws_ids_testbed/bin/send_pcap_to_ids.sh "$PCAP_PATH" "$SCENARIO"; then
        mv "$PCAP_PATH" "$SENT_DIR/$FILE_NAME"
        SENT_COUNT=$((SENT_COUNT + 1))
        echo "[victim-sender] Moved to sent: $SENT_DIR/$FILE_NAME"
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo "[victim-sender] Upload failed. Keeping in pending: $PCAP_PATH"
    fi
done

echo "[victim-sender] Sent count: $SENT_COUNT"
echo "[victim-sender] Failed count: $FAILED_COUNT"

if [[ "$FAILED_COUNT" -gt 0 ]]; then
    exit 1
fi
"""

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def _run_remote_command(ssh_client: object, command: str) -> None:
    """Run one remote command and raise an error if it fails."""
    _stdin, stdout, stderr = ssh_client.exec_command(command, get_pty=True)

    for line in stdout:
        print(line, end="")

    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        error_text = stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Remote command failed with exit code {exit_code}: {command}\n{error_text}"
        )
