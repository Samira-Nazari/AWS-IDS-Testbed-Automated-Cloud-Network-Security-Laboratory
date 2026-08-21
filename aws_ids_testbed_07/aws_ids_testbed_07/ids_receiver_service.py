"""Deploy and manage the small IDS receiver service."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_07.remote_runner import RemoteRunner
from aws_ids_testbed_07.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def deploy_ids_receiver_files(project_root: Path) -> int:
    """Copy only essential IDS receiver files to the IDS EC2 instance.

    This copies:
    - ids_receiver_app.py
    - start_ids_receiver.sh

    It does not copy the full project.
    """
    from scp import SCPClient
    import paramiko

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_host = get_public_host(project_root, "ids")

    receiver_app = project_root / "aws_ids_testbed_07" / "ids_receiver_app.py"
    start_script = project_root / "scripts" / "start_ids_receiver.sh"

    if not receiver_app.exists():
        raise FileNotFoundError(f"Receiver app not found: {receiver_app}")

    if not start_script.exists():
        raise FileNotFoundError(f"Start script not found: {start_script}")

    ssh_client = paramiko.SSHClient()

    # Use known_hosts when possible.
    ssh_client.load_system_host_keys()

    # In this private lab, automatically accept a new EC2 host key.
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {username}@{ids_host} ...")
    ssh_client.connect(
        hostname=ids_host,
        username=username,
        key_filename=str(private_key_path),
        timeout=30,
    )

    try:
        print("Creating IDS receiver folder...")
        _run_remote_command(ssh_client, "mkdir -p /home/ubuntu/aws_ids_testbed")

        print("Uploading IDS receiver files...")
        with SCPClient(ssh_client.get_transport()) as scp_client:
            scp_client.put(
                str(receiver_app),
                "/home/ubuntu/aws_ids_testbed/ids_receiver_app.py",
            )
            scp_client.put(
                str(start_script),
                "/home/ubuntu/aws_ids_testbed/start_ids_receiver.sh",
            )

        print("Making start script executable...")
        _run_remote_command(
            ssh_client,
            "chmod +x /home/ubuntu/aws_ids_testbed/start_ids_receiver.sh",
        )

        print("IDS receiver files deployed.")
        return 0

    finally:
        ssh_client.close()


def ids_start_receiver(project_root: Path) -> int:
    """Start the IDS FastAPI receiver in the background.

    This function does not hard-code the IDS IP.

    It reads the IDS public IP from inventory.yaml, connects by SSH,
    starts the receiver, and verifies the local health endpoint on IDS.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        "mkdir -p /home/ubuntu/aws_ids_testbed/logs && "
        "cd /home/ubuntu/aws_ids_testbed && "
        "if curl --fail --silent --show-error --max-time 3 "
        "http://localhost:8000/health > /dev/null; then "
        "echo '[ids] IDS receiver is already healthy.'; "
        "else "
        "echo '[ids] IDS receiver is not healthy. Starting it now...' && "
        "chmod +x /home/ubuntu/aws_ids_testbed/start_ids_receiver.sh && "
        "nohup /home/ubuntu/aws_ids_testbed/start_ids_receiver.sh "
        "> /home/ubuntu/aws_ids_testbed/logs/ids_receiver.log "
        "2>&1 < /dev/null & "
        "echo $! > /home/ubuntu/aws_ids_testbed/ids_receiver.pid && "
        "sleep 5; "
        "fi && "
        "echo '[ids] Checking receiver health...' && "
        "curl --fail --silent --show-error --max-time 5 "
        "http://localhost:8000/health && echo || "
        "(echo '[ids] Receiver failed to become healthy. Recent log:' && "
        "tail -n 80 /home/ubuntu/aws_ids_testbed/logs/ids_receiver.log && "
        "exit 1)"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=ids_public_host,
        command=command,
    )


def ids_list_received_pcaps(project_root: Path) -> int:
    """List PCAP files received by the IDS receiver.

    This function does not hard-code the IDS IP.

    It reads the IDS public IP from inventory.yaml and lists:

        /home/ubuntu/aws_ids_testbed/input/*.pcap
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        "echo '[ids] Received PCAP files:' && "
        "find /home/ubuntu/aws_ids_testbed/input "
        "-maxdepth 1 -type f -name '*.pcap' "
        "-printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\\n' "
        "| sort"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=ids_public_host,
        command=command,
    )


def _run_remote_command(ssh_client, command: str) -> None:
    """Run a short remote command and raise an error if it fails."""
    _stdin, stdout, stderr = ssh_client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()

    if exit_code != 0:
        error_text = stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote command failed: {command}\n{error_text}")
