"""Generate lab traffic between EC2 instances."""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_06.remote_runner import RemoteRunner
from aws_ids_testbed_06.remote_settings import (
    get_private_ip,
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def generate_benign_http(project_root: Path) -> int:
    """Generate benign HTTP traffic from attacker to victim.

    This function does not hard-code IP addresses.

    It uses:
    - attacker public IP for SSH control
    - victim private IP for internal AWS HTTP traffic
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)

    attacker_host = get_public_host(project_root, "attacker")
    victim_private_ip = get_private_ip(project_root, "victim")

    command = f"ab -n 100 -c 5 http://{victim_private_ip}/"

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=attacker_host,
        command=command,
    )


def generate_traffic_by_code(
    project_root: Path,
    traffic_code: str,
    requests: int | None = None,
    concurrency: int | None = None,
    packet_count: int | None = None,
    port: int = 80,
) -> int:
    """Generate one type of lab traffic from attacker to victim.

    Traffic codes:

        1 = benign HTTP traffic
        2 = controlled DoS HTTP flood pattern
        3 = controlled DoS SYN flood pattern

    This function does not hard-code IP addresses.

    It uses:
    - attacker public IP for SSH control
    - victim private IP for internal AWS traffic
    """
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)

    attacker_host = get_public_host(project_root, "attacker")
    victim_private_ip = get_private_ip(project_root, "victim")

    if traffic_code == "1":
        traffic_name = "benign_http"
        final_requests = requests if requests is not None else 100
        final_concurrency = concurrency if concurrency is not None else 5
        command = f"ab -n {final_requests} -c {final_concurrency} http://{victim_private_ip}/"

    elif traffic_code == "2":
        traffic_name = "dos_http_flood"
        final_requests = requests if requests is not None else 1000
        final_concurrency = concurrency if concurrency is not None else 50
        command = f"ab -n {final_requests} -c {final_concurrency} http://{victim_private_ip}/"

    elif traffic_code == "3":
        traffic_name = "dos_syn_flood"
        final_packet_count = packet_count if packet_count is not None else 1000
        command = f"sudo hping3 -S -p {port} -c {final_packet_count} {victim_private_ip}"

    else:
        print("Invalid traffic code.")
        print("Use: 1=benign_http, 2=dos_http_flood, 3=dos_syn_flood")
        return 1

    print(f"Generating traffic type: {traffic_name}")
    print(f"Victim private IP: {victim_private_ip}")

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    return runner.run_command(
        host=attacker_host,
        command=command,
    )
