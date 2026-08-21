"""Deploy and run attacker-side traffic generation scripts."""

from __future__ import annotations

import shlex
from pathlib import Path

from aws_ids_testbed_07.remote_runner import RemoteRunner
from aws_ids_testbed_07.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def deploy_attacker_traffic(project_root: Path) -> int:
    """Copy the attacker traffic script to the attacker EC2 instance.

    This function does not hard-code the attacker IP.

    It reads the attacker public IP from inventory.yaml and copies:

        scripts/generate_traffic.sh

    to:

        /opt/aws_ids_testbed/bin/generate_traffic.sh
    """
    import paramiko
    from scp import SCPClient

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    attacker_public_host = get_public_host(project_root, "attacker")

    local_script_path = project_root / "scripts" / "generate_traffic.sh"
    remote_bin_dir = "/opt/aws_ids_testbed/bin"
    remote_script_path = f"{remote_bin_dir}/generate_traffic.sh"

    if not private_key_path.exists():
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

    if not local_script_path.exists():
        raise FileNotFoundError(f"Attacker traffic script not found: {local_script_path}")

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {username}@{attacker_public_host} ...")
    ssh_client.connect(
        hostname=attacker_public_host,
        username=username,
        key_filename=str(private_key_path),
        timeout=30,
    )

    try:
        print("Creating attacker traffic folder...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=f"sudo mkdir -p {remote_bin_dir} && sudo chown ubuntu:ubuntu {remote_bin_dir}",
        )

        print("Uploading attacker traffic script...")
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.put(str(local_script_path), remote_script_path)

        print("Making attacker traffic script executable...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=f"chmod 700 {remote_script_path}",
        )

        print("Attacker traffic script deployed.")
        return 0

    finally:
        ssh_client.close()


def verify_attacker_traffic(project_root: Path) -> int:
    """Verify that the attacker traffic script is installed correctly.

    This function does not hard-code the attacker IP.

    It checks three things on the attacker:

    1. The traffic script exists.
    2. The traffic script is executable.
    3. The traffic script has valid bash syntax.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    attacker_public_host = get_public_host(project_root, "attacker")

    script_path = "/opt/aws_ids_testbed/bin/generate_traffic.sh"

    command = (
        f"ls -lh {script_path} && "
        f"test -f {script_path} && "
        f"test -x {script_path} && "
        f"bash -n {script_path} && "
        "echo 'Attacker traffic script is installed, executable, and has valid bash syntax.'"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=attacker_public_host,
        command=command,
    )


def attacker_generate_traffic(
    project_root: Path,
    traffic_code: str,
    requests: int | None = None,
    concurrency: int | None = None,
    packet_count: int | None = None,
    port: int | None = None,
) -> int:
    """Generate traffic from attacker using attacker-side config.

    This function does not hard-code the attacker IP or victim IP.

    It reads the attacker public IP from inventory.yaml.
    The attacker-side script reads victim target information from:

        /opt/aws_ids_testbed/config/attacker.env
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    attacker_public_host = get_public_host(project_root, "attacker")

    command_parts = [
        "/opt/aws_ids_testbed/bin/generate_traffic.sh",
        shlex.quote(traffic_code),
    ]

    if requests is not None:
        command_parts.extend(["--requests", str(requests)])

    if concurrency is not None:
        command_parts.extend(["--concurrency", str(concurrency)])

    if packet_count is not None:
        command_parts.extend(["--packet-count", str(packet_count)])

    if port is not None:
        command_parts.extend(["--port", str(port)])

    command = " ".join(command_parts)

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=attacker_public_host,
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
