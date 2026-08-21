"""Configure attacker-side settings for sending traffic to the victim."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_07.remote_runner import RemoteRunner
from aws_ids_testbed_07.remote_settings import (
    get_private_ip,
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def configure_attacker_victim_url(project_root: Path) -> int:
    """Save victim target settings on the attacker EC2 instance.

    This function runs from your local controller machine.

    It reads:
    - attacker public IP from inventory.yaml, for SSH
    - victim private IP from inventory.yaml, for internal AWS traffic

    Then it writes this persistent config file on the attacker:

        /opt/aws_ids_testbed/config/attacker.env

    The attacker traffic script will later read that file.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)

    attacker_public_host = get_public_host(project_root, "attacker")
    victim_private_ip = get_private_ip(project_root, "victim")
    victim_url = f"http://{victim_private_ip}"

    command = (
        "sudo mkdir -p /opt/aws_ids_testbed/config && "
        "sudo tee /opt/aws_ids_testbed/config/attacker.env > /dev/null <<'EOF'\n"
        f"VICTIM_PRIVATE_IP={victim_private_ip}\n"
        f"VICTIM_URL={victim_url}\n"
        "DEFAULT_HTTP_REQUESTS=100\n"
        "DEFAULT_HTTP_CONCURRENCY=5\n"
        "DEFAULT_DOS_HTTP_REQUESTS=1000\n"
        "DEFAULT_DOS_HTTP_CONCURRENCY=50\n"
        "DEFAULT_SYN_PACKET_COUNT=100\n"
        "DEFAULT_TARGET_PORT=80\n"
        "EOF\n"
        "sudo chmod 644 /opt/aws_ids_testbed/config/attacker.env && "
        "echo 'Attacker victim config saved:' && "
        "cat /opt/aws_ids_testbed/config/attacker.env"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=attacker_public_host,
        command=command,
    )


def attacker_verify_victim_url(project_root: Path) -> int:
    """Verify the attacker victim target config file.

    This function does not hard-code the attacker IP.

    It reads the attacker public IP from inventory.yaml and checks:

        /opt/aws_ids_testbed/config/attacker.env

    This file should contain VICTIM_PRIVATE_IP, VICTIM_URL,
    and default traffic parameters.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    attacker_public_host = get_public_host(project_root, "attacker")

    command = (
        "ls -lh /opt/aws_ids_testbed/config/attacker.env && "
        "cat /opt/aws_ids_testbed/config/attacker.env"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=attacker_public_host,
        command=command,
    )
