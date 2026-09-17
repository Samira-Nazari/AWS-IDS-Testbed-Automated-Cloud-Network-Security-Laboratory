"""Deploy and verify the benign traffic agent."""

from __future__ import annotations

import shlex
from pathlib import Path

from aws_ids_testbed_10.benign_generator_lifecycle import (
    BENIGN_GENERATOR_ROLES,
)
from aws_ids_testbed_10.remote_runner import RemoteRunner
from aws_ids_testbed_10.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)

REMOTE_DIR = "/opt/aws_ids_testbed/benign/bin"
REMOTE_SCRIPT = f"{REMOTE_DIR}/generate_benign_traffic.sh"
REMOTE_LOG_DIR = "/opt/aws_ids_testbed/benign/logs"
REMOTE_PID_FILE = "/opt/aws_ids_testbed/benign/traffic_agent.pid"


def deploy_benign_traffic(project_root: Path) -> int:
    """Upload the benign traffic agent to all five generators."""
    import paramiko
    from scp import SCPClient

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    local_script = project_root / "scripts" / "generate_benign_traffic.sh"

    if not local_script.exists():
        raise FileNotFoundError(f"Missing local script: {local_script}")

    for role in BENIGN_GENERATOR_ROLES:
        host = get_public_host(project_root, role)
        print(f"[benign-traffic] Deploying to {role}...")

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=host,
            username=username,
            key_filename=str(private_key_path),
            timeout=30,
        )

        try:
            _stdin, stdout, stderr = client.exec_command(
                f"sudo mkdir -p {REMOTE_DIR} && "
                f"sudo chown {username}:{username} {REMOTE_DIR}",
                get_pty=True,
            )
            if stdout.channel.recv_exit_status() != 0:
                error = stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(error)

            with SCPClient(client.get_transport()) as scp:
                scp.put(str(local_script), REMOTE_SCRIPT)

            _stdin, stdout, stderr = client.exec_command(
                f"chmod 700 {REMOTE_SCRIPT}",
                get_pty=True,
            )
            if stdout.channel.recv_exit_status() != 0:
                error = stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(error)

            print(f"[benign-traffic] {role} deployed successfully.")

        finally:
            client.close()

    return 0


def verify_benign_traffic(project_root: Path) -> int:
    """Verify the deployed benign traffic agent on all generators."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    command = (
        f"test -f {REMOTE_SCRIPT} && "
        f"test -x {REMOTE_SCRIPT} && "
        f"bash -n {REMOTE_SCRIPT} && "
        "echo 'Benign traffic agent is installed and valid.'"
    )

    overall_status = 0

    for role in BENIGN_GENERATOR_ROLES:
        print(f"[benign-traffic] Verifying {role}...")
        status = runner.run_command(
            host=get_public_host(project_root, role),
            command=command,
        )

        if status != 0:
            overall_status = status

    return overall_status


def diagnose_benign_generators(
    project_root: Path,
    duration_seconds: int,
) -> int:
    """Run every benign generator in the foreground for diagnostics."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    command = f"""
set -u

SCRIPT="/opt/aws_ids_testbed/benign/bin/generate_benign_traffic.sh"
CONFIG="/opt/aws_ids_testbed/benign/config/generator.env"
DURATION_SECONDS={duration_seconds}

if [ ! -r "$CONFIG" ]; then
    echo "[diagnostic] Missing configuration: $CONFIG"
    exit 1
fi

if ! . "$CONFIG"; then
    echo "[diagnostic] Could not load configuration."
    exit 1
fi

ROLE="${{GENERATOR_ROLE:-unknown}}"
VICTIM_IP="${{VICTIM_PRIVATE_IP:-missing}}"

echo "[diagnostic] Role: $ROLE"
echo "[diagnostic] Victim IP: $VICTIM_IP"

if [ ! -x "$SCRIPT" ]; then
    echo "[diagnostic] Script is missing or not executable: $SCRIPT"
    exit 1
fi

if ! bash -n "$SCRIPT"; then
    echo "[diagnostic] Script syntax check failed."
    exit 1
fi

required_tools="timeout"

case "$ROLE" in
    benign_generator_01|benign_generator_03)
        required_tools="$required_tools curl"
        ;;
    benign_generator_02)
        required_tools="$required_tools dig python3"
        ;;
    benign_generator_04)
        required_tools="$required_tools mosquitto_pub mosquitto_sub"
        ;;
    benign_generator_05)
        required_tools="$required_tools ping python3"
        ;;
    *)
        echo "[diagnostic] Unknown generator role: $ROLE"
        exit 1
        ;;
esac

for tool in $required_tools; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "[diagnostic] $tool: OK"
    else
        echo "[diagnostic] $tool: MISSING"
        exit 1
    fi
done

echo "[diagnostic] Running foreground traffic profile..."
set +e
timeout --signal=TERM --kill-after=5 "${{DURATION_SECONDS}}s" "$SCRIPT" \
    --duration-seconds "$DURATION_SECONDS"
status=$?
set -e

if [ "$status" -eq 0 ] || [ "$status" -eq 124 ]; then
    echo "[diagnostic] Foreground profile completed normally."
    exit 0
fi

echo "[diagnostic] Foreground profile failed with exit code: $status"
exit "$status"
""".strip()

    overall_status = 0

    for role in BENIGN_GENERATOR_ROLES:
        print(f"[diagnostic] Testing {role}...")
        status = runner.run_command(
            host=get_public_host(project_root, role),
            command=command,
        )

        if status != 0:
            overall_status = status

    return overall_status


def start_benign_traffic(
    project_root: Path,
    duration_seconds: int,
) -> int:
    """Start all generators for a fixed duration."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    overall_status = 0

    for role in BENIGN_GENERATOR_ROLES:
        host = get_public_host(project_root, role)

        inner_command = (
            f"echo $$ > {shlex.quote(REMOTE_PID_FILE)}; "
            f"timeout --signal=TERM --kill-after=10 "
            f"{duration_seconds}s "
            f"{shlex.quote(REMOTE_SCRIPT)} "
            f"--duration-seconds {duration_seconds}; "
            "status=$?; "
            f"rm -f {shlex.quote(REMOTE_PID_FILE)}; "
            "exit \"$status\""
        )

        command = (
            f"mkdir -p {shlex.quote(REMOTE_LOG_DIR)}; "
            f"if [ -f {shlex.quote(REMOTE_PID_FILE)} ] && "
            f"kill -0 \"$(cat {shlex.quote(REMOTE_PID_FILE)})\" 2>/dev/null; then "
            "echo '[benign-traffic] Already running.'; "
            "else "
            f"rm -f {shlex.quote(REMOTE_PID_FILE)}; "
            f"nohup setsid bash -c {shlex.quote(inner_command)} "
            f"> {shlex.quote(REMOTE_LOG_DIR)}/traffic_agent.log "
            "2>&1 < /dev/null & "
            "for attempt in 1 2 3 4 5; do "
            f"if [ -s {shlex.quote(REMOTE_PID_FILE)} ]; then break; fi; "
            "sleep 0.2; "
            "done; "
            f"if [ -s {shlex.quote(REMOTE_PID_FILE)} ]; then "
            "echo '[benign-traffic] Started.'; "
            f"echo \"PID: $(cat {shlex.quote(REMOTE_PID_FILE)})\"; "
            "else "
            "echo '[benign-traffic] Failed to record traffic-agent PID.'; "
            "exit 1; "
            "fi; "
            "fi"
        )

        print(f"[benign-traffic] Starting {role}...")
        status = runner.run_command(host=host, command=command)

        if status != 0:
            overall_status = status

    return overall_status


def stop_benign_traffic(project_root: Path) -> int:
    """Stop all running benign generators."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    overall_status = 0

    for role in BENIGN_GENERATOR_ROLES:
        host = get_public_host(project_root, role)

        command = (
            f"if [ -f {shlex.quote(REMOTE_PID_FILE)} ]; then "
            f"PID=$(cat {shlex.quote(REMOTE_PID_FILE)}); "
            "kill -TERM -- \"-$PID\" 2>/dev/null || "
            "kill -TERM \"$PID\" 2>/dev/null || true; "
            f"rm -f {shlex.quote(REMOTE_PID_FILE)}; "
            "echo '[benign-traffic] Stopped.'; "
            "else "
            "echo '[benign-traffic] Not running.'; "
            "fi"
        )

        print(f"[benign-traffic] Stopping {role}...")
        status = runner.run_command(host=host, command=command)

        if status != 0:
            overall_status = status

    return overall_status


def benign_traffic_status(project_root: Path) -> int:
    """Show the status of all benign generators."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    overall_status = 0

    for role in BENIGN_GENERATOR_ROLES:
        host = get_public_host(project_root, role)

        command = (
            f"if [ -f {shlex.quote(REMOTE_PID_FILE)} ] && "
            f"kill -0 \"$(cat {shlex.quote(REMOTE_PID_FILE)})\" 2>/dev/null; then "
            "echo '[benign-traffic] Status: running'; "
            f"echo \"PID: $(cat {shlex.quote(REMOTE_PID_FILE)})\"; "
            "else "
            "echo '[benign-traffic] Status: stopped'; "
            "fi; "
            f"tail -n 10 {shlex.quote(REMOTE_LOG_DIR)}/traffic_agent.log "
            "2>/dev/null || true"
        )

        print(f"[benign-traffic] Status for {role}...")
        status = runner.run_command(host=host, command=command)

        if status != 0:
            overall_status = status

    return overall_status
