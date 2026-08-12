"""Run setup scripts on EC2 instances."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_06.remote_runner import RemoteRunner
from aws_ids_testbed_06.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_setup_script_path,
    get_ssh_username,
)


def run_setup_for_role(project_root: Path, role: str) -> int:
    """Run the setup bash script for one EC2 role.

    Args:
        project_root: Main project folder.
        role: One of victim, attacker, or ids.

    Returns:
        Remote setup script exit code. Zero means success.
    """
    # Read SSH username from config.yaml.
    username = get_ssh_username(project_root)

    # Find the local .pem key file.
    private_key_path = get_private_key_path(project_root)

    # Find the public IP or public DNS for the selected EC2 instance.
    host = get_public_host(project_root, role)

    # Find the local setup script for this role.
    script_path = get_setup_script_path(project_root, role)

    # Create the object that knows how to connect by SSH and run scripts.
    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    # Upload the setup script to the EC2 instance and run it.
    return runner.run_script(
        host=host,
        local_script_path=script_path,
    )
