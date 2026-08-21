"""Capture traffic on EC2 instances."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_07.remote_runner import RemoteRunner
from aws_ids_testbed_07.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def capture_benign_http_on_victim(project_root: Path) -> int:
    """Capture benign HTTP traffic on the victim EC2 instance.

    This function does not hard-code the victim public IP.
    It reads the victim public IP from inventory.yaml.

    The remote command creates this PCAP file on victim:

        /opt/aws_ids_testbed/pcap/benign_http_test.pcap
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_host = get_public_host(project_root, "victim")

    command = (
        "sudo timeout 20 "
        "tcpdump -i ens5 "
        "-w /opt/aws_ids_testbed/pcap/benign_http_test.pcap "
        "tcp port 80; "
        "status=$?; "
        "if [ $status -eq 124 ]; then exit 0; else exit $status; fi"
    )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_host,
        command=command,
    )


def verify_benign_http_pcap(project_root: Path) -> int:
    """Verify that the benign HTTP PCAP file exists on victim.

    This function reads the victim public IP from inventory.yaml.
    It checks the PCAP file created by capture_benign_http_on_victim().
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    victim_host = get_public_host(project_root, "victim")

    command = "ls -lh /opt/aws_ids_testbed/pcap/benign_http_test.pcap"

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=victim_host,
        command=command,
    )
