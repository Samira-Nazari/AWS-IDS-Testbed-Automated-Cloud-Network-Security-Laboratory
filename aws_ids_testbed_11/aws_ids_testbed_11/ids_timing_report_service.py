"""Run the read-only IDS timing report generator on AWS."""

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


def generate_ids_timing_reports(
    project_root: Path,
    scenario: str | None = None,
    execution_id: str | None = None,
) -> int:
    """Generate reports for the active scenario execution on IDS."""

    runner = RemoteRunner(
        username=get_ssh_username(project_root),
        private_key_path=get_private_key_path(project_root),
    )
    ids_host = get_public_host(project_root, "ids")
    execution_file = (
        f"{REMOTE_BASE_DIR}/inference/"
        "current_scenario_execution_id.txt"
    )

    command_parts = [
        f"cd {shlex.quote(REMOTE_BASE_DIR)} && "
        f"{shlex.quote(REMOTE_PYTHON)} "
        "-m aws_ids_testbed_11.ids_timing_report"
    ]
    if scenario:
        command_parts.append(
            f"--scenario {shlex.quote(scenario)}"
        )

    if execution_id:
        command_parts.append(
            f"--execution-id {shlex.quote(execution_id)}"
        )
        command = " ".join(command_parts)
    else:
        command = (
            f"EXECUTION_ID=$(tr -d '\\r\\n' "
            f"< {shlex.quote(execution_file)}) && "
            + " ".join(command_parts)
            + ' --execution-id "$EXECUTION_ID"'
        )

    return runner.run_command(
        host=ids_host,
        command=command,
    )
