"""Synchronize victim and benign-generator network configuration."""

from __future__ import annotations

import base64
import shlex
from pathlib import Path

from aws_ids_testbed_08.benign_generator_lifecycle import (
    BENIGN_GENERATOR_ROLES,
)
from aws_ids_testbed_08.remote_runner import RemoteRunner
from aws_ids_testbed_08.remote_settings import (
    get_private_ip,
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


def _render_env(values: dict[str, str]) -> str:
    """Render key-value settings as a shell environment file."""
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _write_remote_env(
    runner: RemoteRunner,
    host: str,
    directory: str,
    path: str,
    content: str,
) -> int:
    """Write an environment file on one remote instance."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

    command = (
        f"sudo mkdir -p {shlex.quote(directory)} && "
        f"echo {shlex.quote(encoded)} | base64 --decode | "
        f"sudo tee {shlex.quote(path)} >/dev/null && "
        f"sudo chmod 644 {shlex.quote(path)} && "
        f"echo 'Saved {path}' && "
        f"sudo cat {shlex.quote(path)}"
    )

    return runner.run_command(host=host, command=command)


def configure_benign_network(project_root: Path) -> int:
    """Save current victim/generator private IP settings remotely."""
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)

    victim_public_host = get_public_host(project_root, "victim")
    victim_private_ip = get_private_ip(project_root, "victim")

    generator_records = []
    for role in BENIGN_GENERATOR_ROLES:
        generator_records.append(
            {
                "role": role,
                "private_ip": get_private_ip(project_root, role),
                "public_host": get_public_host(project_root, role),
            }
        )

    runner = RemoteRunner(
        username=username,
        private_key_path=private_key_path,
    )

    victim_values = {
        "BENIGN_ROLE": "victim",
        "VICTIM_PRIVATE_IP": victim_private_ip,
        "VICTIM_HTTP_PORT": "80",
        "VICTIM_HTTPS_PORT": "443",
        "VICTIM_DNS_PORT": "53",
        "VICTIM_MQTT_PORT": "1883",
        "VICTIM_UDP_PORT": "9999",
        "GENERATOR_COUNT": str(len(generator_records)),
    }

    for index, record in enumerate(generator_records, start=1):
        victim_values[f"GENERATOR_{index:02d}_ROLE"] = record["role"]
        victim_values[f"GENERATOR_{index:02d}_PRIVATE_IP"] = record["private_ip"]

    victim_status = _write_remote_env(
        runner=runner,
        host=victim_public_host,
        directory="/opt/aws_ids_testbed/config",
        path="/opt/aws_ids_testbed/config/benign.env",
        content=_render_env(victim_values),
    )

    if victim_status != 0:
        return victim_status

    for record in generator_records:
        generator_values = {
            "GENERATOR_ROLE": record["role"],
            "GENERATOR_PRIVATE_IP": record["private_ip"],
            "VICTIM_PRIVATE_IP": victim_private_ip,
            "VICTIM_HTTP_PORT": "80",
            "VICTIM_HTTPS_PORT": "443",
            "VICTIM_DNS_PORT": "53",
            "VICTIM_DNS_DOMAIN": "iot.local",
            "VICTIM_MQTT_PORT": "1883",
            "VICTIM_UDP_PORT": "9999",
        }

        generator_status = _write_remote_env(
            runner=runner,
            host=record["public_host"],
            directory="/opt/aws_ids_testbed/benign/config",
            path="/opt/aws_ids_testbed/benign/config/generator.env",
            content=_render_env(generator_values),
        )

        if generator_status != 0:
            print(f"Configuration failed for {record['role']}.")
            return generator_status

    print("[benign-network] Victim and generator configuration synchronized.")
    return 0
