"""Control IDS inference agents remotely from the Slurm CLI."""

from __future__ import annotations

import shlex
from pathlib import Path

from aws_ids_testbed_11.remote_runner import RemoteRunner
from aws_ids_testbed_11.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


REMOTE_BASE_DIR = "/home/ubuntu/aws_ids_testbed"
REMOTE_PYTHON = (
    f"{REMOTE_BASE_DIR}/ids_inference_env/bin/python"
)


def prepare_ids_inference_run(project_root: Path) -> int:
    """Prepare the shared AWS inference workspace for any scenario."""

    runner = RemoteRunner(
        username=get_ssh_username(project_root),
        private_key_path=get_private_key_path(project_root),
    )
    ids_host = get_public_host(project_root, "ids")

    python_code = (
        "from pathlib import Path; "
        "from aws_ids_testbed_11.ids_inference_scenario "
        "import start_inference_run; "
        "paths = start_inference_run("
        f"base_directory=Path({REMOTE_BASE_DIR!r})); "
        "print('[ids-inference] Run prepared:', "
        "paths.current_scenario_execution_id_path."
        "read_text().strip())"
    )

    command = (
        f"cd {shlex.quote(REMOTE_BASE_DIR)} && "
        f"{shlex.quote(REMOTE_PYTHON)} "
        f"-c {shlex.quote(python_code)}"
    )

    return runner.run_command(
        host=ids_host,
        command=command,
    )


def read_ids_current_scenario_execution_id(
    project_root: Path,
) -> str:
    """Read the active scenario execution ID from the IDS instance."""

    import paramiko

    private_key_path = get_private_key_path(project_root)
    ids_host = get_public_host(project_root, "ids")
    username = get_ssh_username(project_root)

    if not private_key_path.exists():
        raise FileNotFoundError(
            f"Private key not found: {private_key_path}"
        )

    remote_path = (
        f"{REMOTE_BASE_DIR}/inference/"
        "current_scenario_execution_id.txt"
    )

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(
        paramiko.AutoAddPolicy()
    )

    print(
        f"Reading current IDS execution ID from "
        f"{username}@{ids_host} ..."
    )

    ssh_client.connect(
        hostname=ids_host,
        username=username,
        key_filename=str(private_key_path),
        timeout=30,
    )

    try:
        with ssh_client.open_sftp() as sftp:
            with sftp.open(remote_path, "r") as remote_file:
                execution_id = (
                    remote_file.read()
                    .decode("utf-8")
                    .strip()
                )
    finally:
        ssh_client.close()

    if not execution_id:
        raise ValueError(
            "IDS scenario execution ID is empty."
        )

    return execution_id


def mark_ids_scenario_input_finished(
    project_root: Path,
) -> int:
    """Tell IDS that no more PCAPs are expected for this run."""

    runner = RemoteRunner(
        username=get_ssh_username(project_root),
        private_key_path=get_private_key_path(project_root),
    )
    ids_host = get_public_host(project_root, "ids")

    python_code = (
        "from pathlib import Path; "
        "from aws_ids_testbed_11.ids_inference_workspace "
        "import build_inference_workspace_paths; "
        "from aws_ids_testbed_11.ids_inference_scenario "
        "import read_scenario_execution_id; "
        "from aws_ids_testbed_11.ids_scenario_lifecycle "
        "import mark_scenario_input_finished; "
        f"base = Path({REMOTE_BASE_DIR!r}); "
        "paths = build_inference_workspace_paths("
        "base_directory=base); "
        "execution_id = read_scenario_execution_id(paths); "
        "marker = mark_scenario_input_finished("
        "base_directory=base, execution_id=execution_id); "
        "print('[ids-inference] Scenario input finished:', marker)"
    )

    command = (
        f"cd {shlex.quote(REMOTE_BASE_DIR)} && "
        f"{shlex.quote(REMOTE_PYTHON)} "
        f"-c {shlex.quote(python_code)}"
    )

    return runner.run_command(
        host=ids_host,
        command=command,
    )


def wait_for_ids_scenario_complete(
    project_root: Path,
    timeout_seconds: int = 1800,
) -> int:
    """Wait until the IDS finishes the current scenario."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    runner = RemoteRunner(
        username=get_ssh_username(project_root),
        private_key_path=get_private_key_path(project_root),
    )
    ids_host = get_public_host(project_root, "ids")

    command = (
        "set -eu; "
        f"BASE={shlex.quote(REMOTE_BASE_DIR)}; "
        'EXECUTION_ID_FILE="$BASE/inference/'
        'current_scenario_execution_id.txt"; '
        f"DEADLINE=$(($(date +%s)+{timeout_seconds})); "
        "while true; do "
        'if [ -f "$EXECUTION_ID_FILE" ]; then '
        'EXECUTION_ID=$(tr -d "\\r\\n" < "$EXECUTION_ID_FILE"); '
        'COMPLETE="$BASE/state/scenario_runs/${EXECUTION_ID}.complete.json"; '
        'if [ -f "$COMPLETE" ]; then '
        'echo "[ids-inference] Scenario completed: $COMPLETE"; '
        'echo "[ids-inference] Master predictions: '
        '$BASE/inference/ids_predictions.csv"; '
        "exit 0; "
        "fi; "
        "fi; "
        'if [ "$(date +%s)" -ge "$DEADLINE" ]; then '
        'echo "[ids-inference] Timed out waiting for completion."; '
        "exit 1; "
        "fi; "
        "sleep 5; "
        "done"
    )

    return runner.run_command(
        host=ids_host,
        command=command,
    )


def verify_ids_master_predictions(
    project_root: Path,
) -> int:
    """Verify that the persistent IDS prediction CSV is usable."""

    runner = RemoteRunner(
        username=get_ssh_username(project_root),
        private_key_path=get_private_key_path(project_root),
    )
    ids_host = get_public_host(project_root, "ids")

    predictions_path = (
        f"{REMOTE_BASE_DIR}/inference/ids_predictions.csv"
    )

    command = (
        "set -eu; "
        f"PREDICTIONS={shlex.quote(predictions_path)}; "
        'if [ ! -s "$PREDICTIONS" ]; then '
        'echo "[ids-inference] Prediction CSV is missing or empty."; '
        "exit 1; "
        "fi; "
        'ROWS=$(($(wc -l < "$PREDICTIONS") - 1)); '
        'echo "[ids-inference] Prediction rows: $ROWS"; '
        'echo "[ids-inference] Prediction CSV: $PREDICTIONS"'
    )

    return runner.run_command(
        host=ids_host,
        command=command,
    )


def start_ids_inference_agents(project_root: Path) -> int:
    """Start the input and inference agents on the IDS instance."""

    runner = RemoteRunner(
        username=get_ssh_username(project_root),
        private_key_path=get_private_key_path(project_root),
    )
    ids_host = get_public_host(project_root, "ids")

    command = (
        "set -eu; "
        f"BASE={REMOTE_BASE_DIR}; "
        f"PYTHON_BIN={REMOTE_PYTHON}; "
        "LOG_DIR=\"$BASE/logs\"; "
        "mkdir -p \"$LOG_DIR\"; "
        "start_agent() { "
        "AGENT_NAME=\"$1\"; "
        "MODULE_NAME=\"$2\"; "
        "PID_FILE=\"$BASE/$AGENT_NAME.pid\"; "
        "LOG_FILE=\"$LOG_DIR/$AGENT_NAME.log\"; "
        "if [ -f \"$PID_FILE\" ] && "
        "kill -0 \"$(cat \"$PID_FILE\")\" 2>/dev/null; then "
        "echo \"[$AGENT_NAME] Already running. PID: "
        "$(cat \"$PID_FILE\")\"; "
        "return 0; "
        "fi; "
        "rm -f \"$PID_FILE\"; "
        "cd \"$BASE\"; "
        "nohup \"$PYTHON_BIN\" -m \"$MODULE_NAME\" "
        "> \"$LOG_FILE\" 2>&1 < /dev/null & "
        "AGENT_PID=$!; "
        "echo \"$AGENT_PID\" > \"$PID_FILE\"; "
        "sleep 2; "
        "if kill -0 \"$AGENT_PID\" 2>/dev/null; then "
        "echo \"[$AGENT_NAME] Started. PID: $AGENT_PID\"; "
        "else "
        "echo \"[$AGENT_NAME] Failed to start.\"; "
        "tail -n 30 \"$LOG_FILE\" 2>/dev/null || true; "
        "rm -f \"$PID_FILE\"; "
        "return 1; "
        "fi; "
        "}; "
        "start_agent "
        "ids_inference_agent "
        "aws_ids_testbed_11.ids_inference_agent; "
        "start_agent "
        "ids_input_agent "
        "aws_ids_testbed_11.ids_input_agent"
    )

    return runner.run_command(
        host=ids_host,
        command=command,
    )


def start_ids_alert_agent(project_root: Path) -> int:
    """Start the standalone IDS alert agent on the IDS instance."""

    runner = RemoteRunner(
        username=get_ssh_username(project_root),
        private_key_path=get_private_key_path(project_root),
    )
    ids_host = get_public_host(project_root, "ids")

    command = (
        "set -eu; "
        f"BASE={shlex.quote(REMOTE_BASE_DIR)}; "
        f"PYTHON_BIN={shlex.quote(REMOTE_PYTHON)}; "
        'LOG_DIR="$BASE/logs"; '
        'PID_FILE="$BASE/ids_alert_agent.pid"; '
        'LOG_FILE="$LOG_DIR/ids_alert_agent.log"; '
        'mkdir -p "$LOG_DIR"; '
        'if [ -f "$PID_FILE" ] && '
        'kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then '
        'echo "[ids-alert-agent] Already running. PID: '
        '$(cat "$PID_FILE")"; '
        'else '
        'rm -f "$PID_FILE"; '
        'cd "$BASE"; '
        'nohup "$PYTHON_BIN" '
        '-m aws_ids_testbed_11.ids_alert_agent '
        '> "$LOG_FILE" 2>&1 < /dev/null & '
        'AGENT_PID=$!; '
        'echo "$AGENT_PID" > "$PID_FILE"; '
        'sleep 2; '
        'if kill -0 "$AGENT_PID" 2>/dev/null; then '
        'echo "[ids-alert-agent] Started. PID: $AGENT_PID"; '
        'else '
        'echo "[ids-alert-agent] Failed to start."; '
        'tail -n 30 "$LOG_FILE" 2>/dev/null || true; '
        'rm -f "$PID_FILE"; '
        'exit 1; '
        'fi; '
        'fi'
    )

    return runner.run_command(
        host=ids_host,
        command=command,
    )


def watch_ids_alerts(project_root: Path) -> int:
    """Stream colored IDS alert events to the Slurm terminal."""

    runner = RemoteRunner(
        username=get_ssh_username(project_root),
        private_key_path=get_private_key_path(project_root),
    )
    ids_host = get_public_host(project_root, "ids")

    events_path = (
        f"{REMOTE_BASE_DIR}/logs/ids_alert_events.jsonl"
    )

    command = (
        "set -eu; "
        f"EVENTS={shlex.quote(events_path)}; "
        'mkdir -p "$(dirname "$EVENTS")"; '
        'touch "$EVENTS"; '
        'tail -n 0 -F "$EVENTS" | '
        'while IFS= read -r line; do '
        'case "$line" in '
        '*\'"verdict": "BENIGN"\'*) '
        'printf "\\033[32m%s\\033[0m\\n" "$line";; '
        '*\'"verdict": "ATTACK"\'*) '
        'printf "\\033[31m%s\\033[0m\\n" "$line";; '
        '*) '
        'printf "\\033[33m%s\\033[0m\\n" "$line";; '
        'esac; '
        'done'
    )

    return runner.run_command(
        host=ids_host,
        command=command,
    )
