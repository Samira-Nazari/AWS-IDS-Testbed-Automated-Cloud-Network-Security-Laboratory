"""Deploy IDS PCAP-to-CSV converter files."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_07.remote_runner import RemoteRunner
from aws_ids_testbed_07.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


REMOTE_BASE_DIR = "/home/ubuntu/aws_ids_testbed"
REMOTE_CONVERTER_DIR = f"{REMOTE_BASE_DIR}/ids_pcap_to_csv/pcap2csv"
REMOTE_BIN_DIR = f"{REMOTE_BASE_DIR}/bin"
REMOTE_AGENT_PATH = f"{REMOTE_BIN_DIR}/pcap_to_csv_agent.sh"


def deploy_ids_pcap_converter(project_root: Path) -> int:
    """Upload the PCAP-to-CSV converter and agent to the IDS EC2 instance."""
    import paramiko
    from scp import SCPClient

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    converter_source_dir = project_root / "ids_pcap_to_csv" / "pcap2csv"
    agent_source_path = project_root / "scripts" / "pcap_to_csv_agent.sh"

    if not private_key_path.exists():
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

    if not converter_source_dir.is_dir():
        raise FileNotFoundError(f"Converter folder not found: {converter_source_dir}")

    if not agent_source_path.exists():
        raise FileNotFoundError(f"Converter agent script not found: {agent_source_path}")

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {username}@{ids_public_host} ...")
    ssh_client.connect(
        hostname=ids_public_host,
        username=username,
        key_filename=str(private_key_path),
        timeout=30,
    )

    try:
        print("Creating IDS PCAP-to-CSV folders...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=(
                f"mkdir -p {REMOTE_CONVERTER_DIR} "
                f"{REMOTE_BIN_DIR} "
                f"{REMOTE_BASE_DIR}/output/csv "
                f"{REMOTE_BASE_DIR}/tmp/pcap_to_csv "
                f"{REMOTE_BASE_DIR}/state/pcap_to_csv/converted "
                f"{REMOTE_BASE_DIR}/state/pcap_to_csv/failed"
            ),
        )

        print("Uploading IDS PCAP-to-CSV converter modules...")
        with SCPClient(ssh_client.get_transport()) as scp:
            for source_path in sorted(converter_source_dir.glob("*.py")):
                scp.put(
                    str(source_path),
                    f"{REMOTE_CONVERTER_DIR}/{source_path.name}",
                )

            print("Uploading IDS PCAP-to-CSV agent...")
            scp.put(str(agent_source_path), REMOTE_AGENT_PATH)

        print("Making IDS PCAP-to-CSV agent executable...")
        _run_remote_command(
            ssh_client=ssh_client,
            command=f"chmod 700 {REMOTE_AGENT_PATH}",
        )

        print("IDS PCAP-to-CSV converter deployed.")
        return 0

    finally:
        ssh_client.close()


def verify_ids_pcap_converter(project_root: Path) -> int:
    """Verify the IDS PCAP-to-CSV converter is installed correctly."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        f"CONVERTER_DIR={REMOTE_CONVERTER_DIR}; "
        f"AGENT={REMOTE_AGENT_PATH}; "
        f"PYTHON_BIN={REMOTE_BASE_DIR}/ids_env/bin/python; "
        "echo '[ids-pcap-to-csv] Checking converter files...'; "
        "test -d \"$CONVERTER_DIR\" && "
        "for FILE in Generating_dataset.py Feature_extraction.py Connectivity_features.py "
        "Layered_features.py Supporting_functions.py Dynamic_features.py "
        "Communication_features.py; do "
        "test -f \"$CONVERTER_DIR/$FILE\" || exit 1; "
        "done; "
        "test -f \"$AGENT\" && "
        "test -x \"$AGENT\" && "
        "bash -n \"$AGENT\" && "
        "echo '[ids-pcap-to-csv] Checking Python dependencies...'; "
        "\"$PYTHON_BIN\" -c \"import dpkt, scapy, tqdm, pandas, numpy, scipy; "
        "print('Python converter dependencies are installed.')\" && "
        "\"$PYTHON_BIN\" -m py_compile \"$CONVERTER_DIR/Generating_dataset.py\" && "
        "echo 'IDS PCAP-to-CSV converter is installed and valid.'"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=ids_public_host,
        command=command,
    )


def start_ids_pcap_converter_agent(project_root: Path) -> int:
    """Start the IDS PCAP-to-CSV converter agent in the background."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        f"SCRIPT={REMOTE_AGENT_PATH}; "
        f"LOG_DIR={REMOTE_BASE_DIR}/logs; "
        "LOG_FILE=$LOG_DIR/pcap_to_csv_agent.log; "
        f"PID_FILE={REMOTE_BASE_DIR}/pcap_to_csv_agent.pid; "
        "mkdir -p \"$LOG_DIR\"; "
        "if [ -f \"$PID_FILE\" ] && kill -0 \"$(cat \"$PID_FILE\")\" 2>/dev/null; then "
        "echo '[pcap-to-csv-agent] Already running.'; "
        "echo \"[pcap-to-csv-agent] PID: $(cat \"$PID_FILE\")\"; "
        "else "
        "nohup \"$SCRIPT\" > \"$LOG_FILE\" 2>&1 < /dev/null & "
        "echo $! > \"$PID_FILE\"; "
        "echo '[pcap-to-csv-agent] Started.'; "
        "echo \"[pcap-to-csv-agent] PID: $(cat \"$PID_FILE\")\"; "
        "sleep 2; "
        "fi; "
        "echo '[pcap-to-csv-agent] Recent log:'; "
        "tail -n 30 \"$LOG_FILE\" 2>/dev/null || true"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=ids_public_host,
        command=command,
    )


def stop_ids_pcap_converter_agent(project_root: Path) -> int:
    """Stop the IDS PCAP-to-CSV converter agent."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        f"PID_FILE={REMOTE_BASE_DIR}/pcap_to_csv_agent.pid; "
        "if [ ! -f \"$PID_FILE\" ]; then "
        "echo '[pcap-to-csv-agent] Not running. PID file not found.'; "
        "exit 0; "
        "fi; "
        "PID=$(cat \"$PID_FILE\"); "
        "if kill -0 \"$PID\" 2>/dev/null; then "
        "kill \"$PID\"; "
        "echo \"[pcap-to-csv-agent] Stopped PID: $PID\"; "
        "else "
        "echo \"[pcap-to-csv-agent] PID file exists, but process is not running: $PID\"; "
        "fi; "
        "rm -f \"$PID_FILE\""
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=ids_public_host,
        command=command,
    )


def ids_pcap_converter_agent_status(project_root: Path) -> int:
    """Show IDS PCAP-to-CSV converter agent status and recent logs."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        f"PID_FILE={REMOTE_BASE_DIR}/pcap_to_csv_agent.pid; "
        f"LOG_FILE={REMOTE_BASE_DIR}/logs/pcap_to_csv_agent.log; "
        "if [ -f \"$PID_FILE\" ] && kill -0 \"$(cat \"$PID_FILE\")\" 2>/dev/null; then "
        "echo '[pcap-to-csv-agent] Status: running'; "
        "echo \"[pcap-to-csv-agent] PID: $(cat \"$PID_FILE\")\"; "
        "else "
        "echo '[pcap-to-csv-agent] Status: stopped'; "
        "fi; "
        "echo '[pcap-to-csv-agent] Recent log:'; "
        "tail -n 50 \"$LOG_FILE\" 2>/dev/null || echo '[pcap-to-csv-agent] No log file yet.'"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=ids_public_host,
        command=command,
    )


def ids_list_csv_files(project_root: Path) -> int:
    """List CSV files produced by the IDS PCAP-to-CSV converter."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        "echo '[ids] Converted CSV files:' && "
        f"find {REMOTE_BASE_DIR}/output/csv "
        "-maxdepth 1 -type f -name '*.csv' "
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


def ids_convert_pcap(project_root: Path, pcap_path: str) -> int:
    """Convert one IDS PCAP file to CSV on the IDS EC2 instance."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        f"PYTHON_BIN={REMOTE_BASE_DIR}/ids_env/bin/python; "
        f"CONVERTER={REMOTE_CONVERTER_DIR}/Generating_dataset.py; "
        f"PCAP_PATH={pcap_path}; "
        f"CSV_OUTPUT_DIR={REMOTE_BASE_DIR}/output/csv; "
        f"WORK_DIR={REMOTE_BASE_DIR}/tmp/pcap_to_csv; "
        "echo '[ids-pcap-to-csv] Converting one PCAP file...'; "
        "echo \"[ids-pcap-to-csv] PCAP: $PCAP_PATH\"; "
        "test -f \"$PCAP_PATH\" && "
        "\"$PYTHON_BIN\" \"$CONVERTER\" "
        "--pcap-file \"$PCAP_PATH\" "
        "--csv-output-directory \"$CSV_OUTPUT_DIR\" "
        "--work-directory \"$WORK_DIR\""
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=ids_public_host,
        command=command,
    )


def ids_convert_received_pcaps(project_root: Path) -> int:
    """Convert all received IDS PCAP files to CSV on the IDS EC2 instance."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    command = (
        f"PYTHON_BIN={REMOTE_BASE_DIR}/ids_env/bin/python; "
        f"CONVERTER={REMOTE_CONVERTER_DIR}/Generating_dataset.py; "
        f"PCAP_INPUT_DIR={REMOTE_BASE_DIR}/input; "
        f"CSV_OUTPUT_DIR={REMOTE_BASE_DIR}/output/csv; "
        f"WORK_DIR={REMOTE_BASE_DIR}/tmp/pcap_to_csv; "
        "echo '[ids-pcap-to-csv] Converting all received PCAP files...'; "
        "\"$PYTHON_BIN\" \"$CONVERTER\" "
        "--pcap-directory \"$PCAP_INPUT_DIR\" "
        "--csv-output-directory \"$CSV_OUTPUT_DIR\" "
        "--work-directory \"$WORK_DIR\""
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=ids_public_host,
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
