"""Deploy and verify the victim continuous capture agent."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_06.remote_runner import RemoteRunner
from aws_ids_testbed_06.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def deploy_victim_capture_agent(project_root: Path) -> int:
    """Copy continuous_capture_agent.sh to the victim EC2 instance."""
    import paramiko
    from scp import SCPClient

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    local_script_path = project_root / "scripts" / "continuous_capture_agent.sh"
    remote_bin_dir = "/opt/aws_ids_testbed/bin"
    remote_script_path = f"{remote_bin_dir}/continuous_capture_agent.sh"

    if not private_key_path.exists():
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

    if not local_script_path.exists():
        raise FileNotFoundError(f"Capture agent script not found: {local_script_path}")

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
        print("Creating victim agent folder...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=(
                f"sudo mkdir -p {remote_bin_dir} && "
                f"sudo chown ubuntu:ubuntu {remote_bin_dir}"
            ),
        )

        print("Uploading victim capture agent...")
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.put(str(local_script_path), remote_script_path)

        print("Making victim capture agent executable...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=f"chmod 700 {remote_script_path}",
        )

        print("Victim capture agent deployed.")
        return 0

    finally:
        ssh_client.close()


def verify_victim_capture_agent(project_root: Path) -> int:
    """Verify the victim continuous capture agent exists and is executable."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    script_path = "/opt/aws_ids_testbed/bin/continuous_capture_agent.sh"

    command = (
        f"ls -lh {script_path} && "
        f"test -f {script_path} && "
        f"test -x {script_path} && "
        f"bash -n {script_path} && "
        "echo 'Victim capture agent is installed, executable, and has valid bash syntax.'"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def start_victim_capture_agent(project_root: Path) -> int:
    """Start the victim continuous capture agent in the background."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = (
        "SCRIPT=/opt/aws_ids_testbed/bin/continuous_capture_agent.sh; "
        "LOG_DIR=/opt/aws_ids_testbed/logs; "
        "LOG_FILE=$LOG_DIR/continuous_capture_agent.log; "
        "PID_FILE=/opt/aws_ids_testbed/continuous_capture_agent.pid; "
        "mkdir -p \"$LOG_DIR\"; "
        "if [ -f \"$PID_FILE\" ] && kill -0 \"$(cat \"$PID_FILE\")\" 2>/dev/null; then "
        "echo '[capture-agent] Already running.'; "
        "echo \"[capture-agent] PID: $(cat \"$PID_FILE\")\"; "
        "else "
        "nohup \"$SCRIPT\" > \"$LOG_FILE\" 2>&1 < /dev/null & "
        "echo $! > \"$PID_FILE\"; "
        "echo '[capture-agent] Started.'; "
        "echo \"[capture-agent] PID: $(cat \"$PID_FILE\")\"; "
        "sleep 2; "
        "fi; "
        "echo '[capture-agent] Recent log:'; "
        "tail -n 20 \"$LOG_FILE\" 2>/dev/null || true"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def stop_victim_capture_agent(project_root: Path) -> int:
    """Stop the victim continuous capture agent."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = (
        "PID_FILE=/opt/aws_ids_testbed/continuous_capture_agent.pid; "
        "if [ ! -f \"$PID_FILE\" ]; then "
        "echo '[capture-agent] Not running. PID file not found.'; "
        "exit 0; "
        "fi; "
        "PID=$(cat \"$PID_FILE\"); "
        "if kill -0 \"$PID\" 2>/dev/null; then "
        "kill \"$PID\"; "
        "echo \"[capture-agent] Stopped PID: $PID\"; "
        "else "
        "echo \"[capture-agent] PID file exists, but process is not running: $PID\"; "
        "fi; "
        "rm -f \"$PID_FILE\""
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def victim_capture_agent_status(project_root: Path) -> int:
    """Show victim continuous capture agent status and recent logs."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = (
        "PID_FILE=/opt/aws_ids_testbed/continuous_capture_agent.pid; "
        "LOG_FILE=/opt/aws_ids_testbed/logs/continuous_capture_agent.log; "
        "if [ -f \"$PID_FILE\" ] && kill -0 \"$(cat \"$PID_FILE\")\" 2>/dev/null; then "
        "echo '[capture-agent] Status: running'; "
        "echo \"[capture-agent] PID: $(cat \"$PID_FILE\")\"; "
        "else "
        "echo '[capture-agent] Status: stopped'; "
        "fi; "
        "echo '[capture-agent] Recent log:'; "
        "tail -n 30 \"$LOG_FILE\" 2>/dev/null || echo '[capture-agent] No log file yet.'"
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
