"""Configure victim-side settings for sending PCAP files to IDS."""

from __future__ import annotations

import shlex
from pathlib import Path

from aws_ids_testbed_06.remote_runner import RemoteRunner
from aws_ids_testbed_06.remote_settings import (
    get_private_ip,
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def configure_victim_ids_url(project_root: Path) -> int:
    """Save IDS receiver settings on the victim EC2 instance.

    This function runs from your local controller machine.

    It reads:
    - victim public IP from inventory.yaml, for SSH
    - IDS private IP from inventory.yaml, for private AWS communication

    Then it writes this persistent config file on the victim:

        /opt/aws_ids_testbed/config/victim.env

    The victim sender script will later read that file.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)

    victim_public_host = get_public_host(project_root, "victim")
    ids_private_ip = get_private_ip(project_root, "ids")

    ids_receiver_url = f"http://{ids_private_ip}:8000"

    command = (
        "sudo mkdir -p /opt/aws_ids_testbed/config && "
        "sudo tee /opt/aws_ids_testbed/config/victim.env > /dev/null <<'EOF'\n"
        f"IDS_RECEIVER_URL={ids_receiver_url}\n"
        "CAPTURE_INTERFACE=ens5\n"
        "CAPTURE_SECONDS=20\n"
        "CAPTURE_TIMEZONE=America/Toronto\n"
        "CAPTURE_ENABLED=true\n"
        "ACTIVE_SCENARIO=benign_http\n"
        "CAPTURE_ROTATE_SECONDS=10\n"
        "MIN_PCAP_BYTES=100\n"
        "EOF\n"
        "sudo chmod 644 /opt/aws_ids_testbed/config/victim.env && "
        "echo 'Victim IDS config saved:' && "
        "cat /opt/aws_ids_testbed/config/victim.env"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def victim_verify_ids_url(project_root: Path) -> int:
    """Verify the victim IDS receiver config file.

    This function does not hard-code the victim IP.

    It reads the victim public IP from inventory.yaml and checks:

        /opt/aws_ids_testbed/config/victim.env

    This file should contain IDS_RECEIVER_URL and capture settings.
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    command = (
        "ls -lh /opt/aws_ids_testbed/config/victim.env && "
        "cat /opt/aws_ids_testbed/config/victim.env"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )


def victim_set_scenario(project_root: Path, scenario: str) -> int:
    """Set the active traffic scenario label on the victim.

    The continuous capture agent will later read ACTIVE_SCENARIO
    from /opt/aws_ids_testbed/config/victim.env.

    This label is used for PCAP filenames, for example:

        dos_http_flood_YYYYMMDD_HHMMSS.pcap
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_public_host = get_public_host(project_root, "victim")

    safe_scenario = shlex.quote(scenario)

    command = (
        "CONFIG_FILE=/opt/aws_ids_testbed/config/victim.env; "
        "if [ ! -f \"$CONFIG_FILE\" ]; then "
        "echo 'victim.env not found. Run configure-victim-ids-url first.'; "
        "exit 1; "
        "fi; "
        f"SCENARIO={safe_scenario}; "
        "if grep -q '^ACTIVE_SCENARIO=' \"$CONFIG_FILE\"; then "
        "sudo sed -i \"s/^ACTIVE_SCENARIO=.*/ACTIVE_SCENARIO=$SCENARIO/\" \"$CONFIG_FILE\"; "
        "else "
        "echo \"ACTIVE_SCENARIO=$SCENARIO\" | sudo tee -a \"$CONFIG_FILE\" > /dev/null; "
        "fi; "
        "echo 'Victim active scenario saved:'; "
        "grep '^ACTIVE_SCENARIO=' \"$CONFIG_FILE\""
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_public_host,
        command=command,
    )
