"""Deploy and manage victim-side traffic capture scripts."""

from __future__ import annotations

import shlex
from pathlib import Path

from aws_ids_testbed_07.remote_runner import RemoteRunner
from aws_ids_testbed_07.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def deploy_victim_capture(project_root: Path) -> int:
    """Copy the victim capture script to the victim EC2 instance.

    This function does not hard-code the victim IP.

    It reads the victim public IP from inventory.yaml and copies:

        scripts/capture_scenario.sh

    to:

        /opt/aws_ids_testbed/bin/capture_scenario.sh
    """
    import paramiko
    from scp import SCPClient

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    local_capture_path = project_root / "scripts" / "capture_scenario.sh"
    remote_bin_dir = "/opt/aws_ids_testbed/bin"
    remote_capture_path = f"{remote_bin_dir}/capture_scenario.sh"

    if not private_key_path.exists():
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

    if not local_capture_path.exists():
        raise FileNotFoundError(f"Victim capture script not found: {local_capture_path}")

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {username}@{victim_public_host} ...")
    ssh_client.connect(
        hostname=victim_public_host,
        username=username,
        key_filename=str(private_key_path),
        timeout=30,
    )

    try:
        print("Creating victim capture folder...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=f"sudo mkdir -p {remote_bin_dir} && sudo chown ubuntu:ubuntu {remote_bin_dir}",
        )

        print("Uploading victim capture script...")
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.put(str(local_capture_path), remote_capture_path)

        print("Making victim capture script executable...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=f"chmod 700 {remote_capture_path}",
        )

        print("Victim capture script deployed.")
        return 0

    finally:
        ssh_client.close()


def verify_victim_capture(project_root: Path) -> int:
    """Verify that the victim capture script is installed correctly.

    This function does not hard-code the victim IP.

    It checks three things on the victim:

    1. The capture script exists.
    2. The capture script is executable.
    3. The capture script has valid bash syntax.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    capture_path = "/opt/aws_ids_testbed/bin/capture_scenario.sh"

    command = (
        f"ls -lh {capture_path} && "
        f"test -f {capture_path} && "
        f"test -x {capture_path} && "
        f"bash -n {capture_path} && "
        "echo 'Victim capture script is installed, executable, and has valid bash syntax.'"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def victim_capture_scenario(
    project_root: Path,
    scenario: str,
    seconds: int,
) -> int:
    """Capture one traffic scenario on the victim EC2 instance.

    This function does not hard-code the victim IP.

    It reads the victim public IP from inventory.yaml.
    Then it runs:

        /opt/aws_ids_testbed/bin/capture_scenario.sh SCENARIO SECONDS

    on the victim.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = (
        "/opt/aws_ids_testbed/bin/capture_scenario.sh "
        f"{shlex.quote(scenario)} "
        f"{seconds}"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def victim_list_pcaps(project_root: Path) -> int:
    """List victim PCAP files in writing, pending, sent, and failed folders.

    This function does not hard-code the victim IP.

    It reads the victim public IP from inventory.yaml.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = (
        "CONFIG_FILE=/opt/aws_ids_testbed/config/victim.env; "
        "if [ -f \"$CONFIG_FILE\" ]; then . \"$CONFIG_FILE\"; fi; "
        "TIMEZONE=\"${CAPTURE_TIMEZONE:-America/Toronto}\"; "
        "BASE_DIR=/opt/aws_ids_testbed/pcap; "
        "echo \"[victim] Listing times in timezone: $TIMEZONE\"; "
        "for STATUS_DIR in writing pending sent failed; do "
        "echo \"[victim] $STATUS_DIR PCAP files:\"; "
        "TZ=\"$TIMEZONE\" find \"$BASE_DIR/$STATUS_DIR\" -maxdepth 1 -type f "
        "\\( -name '*.pcap' -o -name '*.pcap.tmp' \\) "
        "-printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\\n' 2>/dev/null "
        "| sort; "
        "done"
    )

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
