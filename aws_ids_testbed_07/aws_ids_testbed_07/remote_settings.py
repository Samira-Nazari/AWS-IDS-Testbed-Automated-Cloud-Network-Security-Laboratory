"""Prepare SSH and setup-script settings for remote EC2 setup.

This file does not connect to AWS.
This file does not create EC2 instances.
This file only reads local project files and returns useful values.

It reads:

- config.yaml
- inventory.yaml
- scripts/setup_victim.sh
- scripts/setup_attacker.sh
- scripts/setup_ids.sh
"""

from __future__ import annotations

from pathlib import Path

from aws_ids_testbed_07.config import load_config
from aws_ids_testbed_07.inventory import load_inventory


def get_ssh_username(project_root: Path) -> str:
    """Return the SSH username from config.yaml.

    In our Ubuntu EC2 instances, the SSH username is usually:

        ubuntu

    This function first reads config.yaml.
    If config.yaml has ssh.username, it returns that value.
    If not, it safely returns the default value: ubuntu.
    """
    # Load all values from config.yaml.
    config = load_config(project_root)

    # Read only the ssh section.
    # Example:
    # ssh:
    #   username: ubuntu
    #   private_key_path: null
    ssh_config = config.get("ssh", {})

    # Return the username from config.yaml.
    # If it is missing, use "ubuntu".
    return ssh_config.get("username", "ubuntu")


def get_private_key_path(project_root: Path) -> Path:
    """Return the local .pem private key path.

    The .pem file is needed for SSH login.

    There are two possible ways to find it:

    1. If config.yaml has ssh.private_key_path, use that exact path.
    2. Otherwise, use aws.key_name and assume the file is in the project root.

    Example:

        aws.key_name = aws-ids-testbed-key

    Then the expected local file is:

        aws-ids-testbed-key.pem
    """
    # Load all values from config.yaml.
    config = load_config(project_root)

    # Read the ssh section for private_key_path.
    ssh_config = config.get("ssh", {})

    # Read the aws section for key_name.
    aws_config = config.get("aws", {})

    # If the user wrote a private key path in config.yaml, use it.
    configured_path = ssh_config.get("private_key_path")
    if configured_path:
        return Path(configured_path).expanduser()

    # If private_key_path is null, build the default key path.
    # Example:
    # project_root / "aws-ids-testbed-key.pem"
    key_name = aws_config.get("key_name", "aws-ids-testbed-key")
    return project_root / f"{key_name}.pem"


def get_public_host(project_root: Path, role: str) -> str:
    """Return the public IP or public DNS name for one EC2 role.

    The role must be one of:

        victim
        attacker
        ids

    This function reads inventory.yaml.

    inventory.yaml is created by our create-* and refresh-* commands.
    It stores information such as:

        instance_id
        state
        public_ip
        private_ip
        public_dns

    We need public_ip or public_dns because SSH from your computer uses
    the public address of the EC2 instance.
    """
    # Load saved EC2 information from inventory.yaml.
    inventory = load_inventory(project_root)

    # Get the dictionary for one role.
    # Example:
    # inventory["instances"]["victim"]
    instance_data = inventory.get("instances", {}).get(role)

    # If the role is not found, the instance may not have been created yet.
    if not instance_data:
        raise ValueError(f"No inventory found for role: {role}")

    # Prefer public_ip. If it is missing, try public_dns.
    public_host = instance_data.get("public_ip") or instance_data.get("public_dns")

    # If both are missing, the instance may not be running or refreshed yet.
    if not public_host:
        raise ValueError(f"No public IP or public DNS found for role: {role}")

    return public_host


def get_private_ip(project_root: Path, role: str) -> str:
    """Return the private IP address for one EC2 role.

    The private IP is used for traffic inside the AWS VPC.

    Example:

        attacker sends HTTP traffic to victim private IP

    This avoids sending experiment traffic over the public Internet.
    """
    # Load saved EC2 information from inventory.yaml.
    inventory = load_inventory(project_root)

    # Get the dictionary for one role.
    instance_data = inventory.get("instances", {}).get(role)

    # If the role is not found, the instance may not have been created yet.
    if not instance_data:
        raise ValueError(f"No inventory found for role: {role}")

    private_ip = instance_data.get("private_ip")

    # If private_ip is missing, refresh the instance first.
    if not private_ip:
        raise ValueError(f"No private IP found for role: {role}")

    return private_ip


def get_setup_script_path(project_root: Path, role: str) -> Path:
    """Return the setup bash script path for one EC2 role.

    Example:

        role = victim

    returns:

        scripts/setup_victim.sh

    This only returns the local file path.
    It does not run the script.
    """
    return project_root / "scripts" / f"setup_{role}.sh"
