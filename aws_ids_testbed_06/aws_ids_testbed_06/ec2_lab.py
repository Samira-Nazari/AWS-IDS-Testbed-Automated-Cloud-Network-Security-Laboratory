"""EC2 lab helpers for AWS IDS Testbed 03."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aws_ids_testbed_06.config import load_config


def get_aws_region(project_root: Path) -> str:
    """Return the AWS region from config.yaml."""
    config = load_config(project_root)
    return config.get("aws_region", "us-east-1")


def get_aws_settings(project_root: Path) -> dict[str, Any]:
    """Return the aws section from config.yaml."""
    config = load_config(project_root)
    return config.get("aws", {})


def get_instance_settings(project_root: Path) -> dict[str, Any]:
    """Return the instances section from config.yaml."""
    config = load_config(project_root)
    return config.get("instances", {})


def create_ec2_client(project_root: Path):
    """Create a boto3 EC2 client.

    This function only creates a local boto3 client object.
    It does not launch EC2 instances by itself.
    """
    import boto3

    region = get_aws_region(project_root)
    return boto3.client("ec2", region_name=region)


def create_one_instance(project_root: Path, role: str) -> dict[str, Any]:
    """Create one EC2 instance for victim, attacker, or ids.

    This function sends a real request to AWS when it is called.
    Do not call it until config.yaml has real AWS values.
    """
    # Read the shared AWS settings from config.yaml.
    # These settings are the same for all three EC2 instances.
    aws_settings = get_aws_settings(project_root)

    # Read the instance settings from config.yaml.
    # This section contains victim, attacker, and ids blocks.
    instance_settings = get_instance_settings(project_root)

    # Select only the settings for the requested role.
    # Example: role="victim" reads instances.victim from config.yaml.
    role_settings = instance_settings[role]

    # Create the boto3 EC2 client.
    # The client is the object that talks to the AWS EC2 API.
    ec2 = create_ec2_client(project_root)

    # Ask AWS to create one EC2 instance.
    # MinCount=1 and MaxCount=1 means create exactly one machine.
    response = ec2.run_instances(
        ImageId=role_settings["ami_id"],
        InstanceType=role_settings["instance_type"],
        KeyName=aws_settings["key_name"],
        SecurityGroupIds=[aws_settings["security_group_id"]],
        SubnetId=aws_settings["subnet_id"],
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    # This name appears in the AWS EC2 console.
                    {"Key": "Name", "Value": role_settings["name"]},
                    # This helps us remember the job of the instance.
                    {"Key": "Role", "Value": role},
                    # This helps us find resources belonging to this project.
                    {"Key": "Project", "Value": "aws-ids-testbed"},
                ],
            }
        ],
    )

    # AWS returns a large response. We only need the first created instance.
    instance = response["Instances"][0]

    # Return a small dictionary that is easy to print and later save.
    return {
        "role": role,
        "name": role_settings["name"],
        "instance_id": instance["InstanceId"],
        "state": instance["State"]["Name"],
    }


def refresh_instance_info(project_root: Path, instance_id: str) -> dict[str, Any]:
    """Ask AWS for the latest information about one EC2 instance.

    This is useful after instance creation because public/private IP addresses
    may not be ready in the first create response.
    """
    ec2 = create_ec2_client(project_root)

    # describe_instances asks AWS to return details for the given instance ID.
    response = ec2.describe_instances(InstanceIds=[instance_id])

    # AWS returns reservations, and each reservation contains instances.
    instance = response["Reservations"][0]["Instances"][0]

    return {
        "instance_id": instance["InstanceId"],
        "state": instance["State"]["Name"],
        "public_ip": instance.get("PublicIpAddress"),
        "private_ip": instance.get("PrivateIpAddress"),
        "public_dns": instance.get("PublicDnsName"),
    }


def terminate_instance(project_root: Path, instance_id: str) -> dict[str, Any]:
    """Terminate one EC2 instance by instance ID.

    This function sends a real terminate request to AWS when it is called.
    """
    ec2 = create_ec2_client(project_root)

    # Ask AWS to terminate the selected EC2 instance.
    response = ec2.terminate_instances(InstanceIds=[instance_id])

    instance_state = response["TerminatingInstances"][0]

    return {
        "instance_id": instance_state["InstanceId"],
        "previous_state": instance_state["PreviousState"]["Name"],
        "state": instance_state["CurrentState"]["Name"],
    }
