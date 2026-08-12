"""Simple command line interface for AWS IDS Testbed 06."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from aws_ids_testbed_06.attacker_config_service import (
    attacker_verify_victim_url,
    configure_attacker_victim_url,
)
from aws_ids_testbed_06.attacker_traffic_service import (
    attacker_generate_traffic,
    deploy_attacker_traffic,
    verify_attacker_traffic,
)
from aws_ids_testbed_06.capture_service import (
    capture_benign_http_on_victim,
    verify_benign_http_pcap,
)
from aws_ids_testbed_06.config import load_config
from aws_ids_testbed_06.ec2_lab import (
    create_one_instance,
    refresh_instance_info,
    terminate_instance,
)
from aws_ids_testbed_06.ids_receiver_service import (
    deploy_ids_receiver_files,
    ids_list_received_pcaps,
    ids_start_receiver,
)
from aws_ids_testbed_06.inventory import load_inventory, update_instance
from aws_ids_testbed_06.setup_service import run_setup_for_role
from aws_ids_testbed_06.traffic_service import (
    generate_benign_http,
    generate_traffic_by_code,
)
from aws_ids_testbed_06.victim_capture_service import (
    deploy_victim_capture,
    verify_victim_capture,
    victim_capture_scenario,
    victim_list_pcaps,
)
from aws_ids_testbed_06.victim_capture_agent_service import (
    deploy_victim_capture_agent,
    start_victim_capture_agent,
    stop_victim_capture_agent,
    victim_capture_agent_status,
    verify_victim_capture_agent,
)
from aws_ids_testbed_06.victim_config_service import (
    configure_victim_ids_url,
    victim_set_scenario,
    victim_verify_ids_url,
)
from aws_ids_testbed_06.victim_sender_service import (
    deploy_victim_sender,
    verify_victim_sender,
    victim_send_pcap,
    victim_send_pending_pcaps,
)
from aws_ids_testbed_06.victim_sender_agent_service import (
    deploy_victim_sender_agent,
    start_victim_sender_agent,
    stop_victim_sender_agent,
    victim_sender_agent_status,
    verify_victim_sender_agent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def status(_: argparse.Namespace) -> int:
    """Show basic project status."""
    print("AWS IDS Testbed 06")
    print(f"Project root: {PROJECT_ROOT}")
    print("Status: package created")
    return 0


def show_config(_: argparse.Namespace) -> int:
    """Show values loaded from config.yaml."""
    config = load_config(PROJECT_ROOT)
    print(yaml.safe_dump(config, sort_keys=False).rstrip())
    return 0


def show_inventory(_: argparse.Namespace) -> int:
    """Show saved EC2 inventory information."""
    inventory = load_inventory(PROJECT_ROOT)

    if not inventory:
        print("inventory.yaml is empty or does not exist yet.")
        return 0

    print(yaml.safe_dump(inventory, sort_keys=False).rstrip())
    return 0


def create_victim(_: argparse.Namespace) -> int:
    """Create the victim EC2 instance."""
    result = create_one_instance(PROJECT_ROOT, "victim")
    update_instance(PROJECT_ROOT, "victim", result)
    print(yaml.safe_dump(result, sort_keys=False).rstrip())
    return 0


def create_attacker(_: argparse.Namespace) -> int:
    """Create the attacker EC2 instance."""
    result = create_one_instance(PROJECT_ROOT, "attacker")
    update_instance(PROJECT_ROOT, "attacker", result)
    print(yaml.safe_dump(result, sort_keys=False).rstrip())
    return 0


def create_ids(_: argparse.Namespace) -> int:
    """Create the IDS EC2 instance."""
    result = create_one_instance(PROJECT_ROOT, "ids")
    update_instance(PROJECT_ROOT, "ids", result)
    print(yaml.safe_dump(result, sort_keys=False).rstrip())
    return 0


def refresh_role(role: str) -> int:
    """Refresh one EC2 role from AWS and save it into inventory.yaml."""
    inventory = load_inventory(PROJECT_ROOT)
    instance_data = inventory.get("instances", {}).get(role)

    if not instance_data:
        print(f"No inventory found for role: {role}")
        return 1

    instance_id = instance_data["instance_id"]
    refreshed = refresh_instance_info(PROJECT_ROOT, instance_id)

    # Keep the original role and name, then add the fresh AWS details.
    updated = {**instance_data, **refreshed}

    update_instance(PROJECT_ROOT, role, updated)
    print(yaml.safe_dump(updated, sort_keys=False).rstrip())
    return 0


def refresh_victim(_: argparse.Namespace) -> int:
    """Refresh victim EC2 information."""
    return refresh_role("victim")


def refresh_attacker(_: argparse.Namespace) -> int:
    """Refresh attacker EC2 information."""
    return refresh_role("attacker")


def refresh_ids(_: argparse.Namespace) -> int:
    """Refresh IDS EC2 information."""
    return refresh_role("ids")


def terminate_role(role: str) -> int:
    """Terminate one EC2 role using the instance ID saved in inventory.yaml."""
    inventory = load_inventory(PROJECT_ROOT)
    instance_data = inventory.get("instances", {}).get(role)

    if not instance_data:
        print(f"No inventory found for role: {role}")
        return 1

    instance_id = instance_data["instance_id"]
    terminated = terminate_instance(PROJECT_ROOT, instance_id)

    # Keep the role/name and record the latest termination state.
    updated = {**instance_data, **terminated}

    update_instance(PROJECT_ROOT, role, updated)
    print(yaml.safe_dump(updated, sort_keys=False).rstrip())
    return 0


def terminate_victim(_: argparse.Namespace) -> int:
    """Terminate the victim EC2 instance."""
    return terminate_role("victim")


def terminate_attacker(_: argparse.Namespace) -> int:
    """Terminate the attacker EC2 instance."""
    return terminate_role("attacker")


def terminate_ids(_: argparse.Namespace) -> int:
    """Terminate the IDS EC2 instance."""
    return terminate_role("ids")


def setup_victim(_: argparse.Namespace) -> int:
    """Run setup_victim.sh on the victim EC2 instance."""
    return run_setup_for_role(PROJECT_ROOT, "victim")


def setup_attacker(_: argparse.Namespace) -> int:
    """Run setup_attacker.sh on the attacker EC2 instance."""
    return run_setup_for_role(PROJECT_ROOT, "attacker")


def setup_ids(_: argparse.Namespace) -> int:
    """Run setup_ids.sh on the IDS EC2 instance."""
    return run_setup_for_role(PROJECT_ROOT, "ids")


def capture_benign_http(_: argparse.Namespace) -> int:
    """Capture benign HTTP traffic on the victim EC2 instance."""
    return capture_benign_http_on_victim(PROJECT_ROOT)


def verify_benign_pcap(_: argparse.Namespace) -> int:
    """Verify benign HTTP PCAP file exists on victim."""
    return verify_benign_http_pcap(PROJECT_ROOT)


def generate_benign_http_traffic(_: argparse.Namespace) -> int:
    """Generate benign HTTP traffic from attacker to victim."""
    return generate_benign_http(PROJECT_ROOT)


def generate_traffic_command(args: argparse.Namespace) -> int:
    """Generate selected lab traffic from attacker to victim."""
    return generate_traffic_by_code(
        project_root=PROJECT_ROOT,
        traffic_code=args.traffic_code,
        requests=args.requests,
        concurrency=args.concurrency,
        packet_count=args.packet_count,
        port=args.port,
    )


def configure_attacker_victim(_: argparse.Namespace) -> int:
    """Save victim target URL on the attacker EC2 instance."""
    return configure_attacker_victim_url(PROJECT_ROOT)


def attacker_verify_victim_url_command(_: argparse.Namespace) -> int:
    """Verify victim target URL saved on the attacker EC2 instance."""
    return attacker_verify_victim_url(PROJECT_ROOT)


def deploy_attacker_traffic_command(_: argparse.Namespace) -> int:
    """Copy the attacker traffic script to the attacker EC2 instance."""
    return deploy_attacker_traffic(PROJECT_ROOT)


def verify_attacker_traffic_command(_: argparse.Namespace) -> int:
    """Verify the attacker traffic script on the attacker EC2 instance."""
    return verify_attacker_traffic(PROJECT_ROOT)


def attacker_generate_traffic_command(args: argparse.Namespace) -> int:
    """Generate traffic from attacker using attacker-side config."""
    return attacker_generate_traffic(
        project_root=PROJECT_ROOT,
        traffic_code=args.traffic_code,
        requests=args.requests,
        concurrency=args.concurrency,
        packet_count=args.packet_count,
        port=args.port,
    )


def run_scenario_command(args: argparse.Namespace) -> int:
    """Run one labeled scenario with coordinated victim and attacker actions."""
    scenario_by_code = {
        "1": "benign_http",
        "2": "dos_http_flood",
        "3": "dos_syn_flood",
    }

    scenario = scenario_by_code[args.traffic_code]

    print(f"Scenario: {scenario}")
    print("Step 1: Set victim ACTIVE_SCENARIO.")
    scenario_status = victim_set_scenario(
        project_root=PROJECT_ROOT,
        scenario=scenario,
    )
    if scenario_status != 0:
        return scenario_status

    print("Step 2: Start victim capture and sender agents.")
    agents_status = victim_start_agents_command(args)
    if agents_status != 0:
        return agents_status

    print("Step 3: Generate matching traffic from attacker.")
    traffic_status = attacker_generate_traffic(
        project_root=PROJECT_ROOT,
        traffic_code=args.traffic_code,
        requests=args.requests,
        concurrency=args.concurrency,
        packet_count=args.packet_count,
        port=args.port,
    )
    if traffic_status != 0:
        return traffic_status

    if args.auto_stop_seconds is not None:
        print(f"Step 4: Wait {args.auto_stop_seconds} seconds before stopping agents.")
        time.sleep(args.auto_stop_seconds)

        print("Step 5: Stop victim capture and sender agents.")
        return victim_stop_agents_command(args)

    return 0


def deploy_ids_receiver(_: argparse.Namespace) -> int:
    """Copy essential IDS receiver files to the IDS EC2 instance."""
    return deploy_ids_receiver_files(PROJECT_ROOT)


def ids_start_receiver_command(_: argparse.Namespace) -> int:
    """Start the IDS FastAPI receiver on the IDS EC2 instance."""
    return ids_start_receiver(PROJECT_ROOT)


def ids_list_received_pcaps_command(_: argparse.Namespace) -> int:
    """List PCAP files received on the IDS EC2 instance."""
    return ids_list_received_pcaps(PROJECT_ROOT)


def configure_victim_ids(_: argparse.Namespace) -> int:
    """Save IDS receiver URL on the victim EC2 instance."""
    return configure_victim_ids_url(PROJECT_ROOT)


def victim_verify_ids_url_command(_: argparse.Namespace) -> int:
    """Verify IDS receiver URL saved on the victim EC2 instance."""
    return victim_verify_ids_url(PROJECT_ROOT)


def victim_set_scenario_command(args: argparse.Namespace) -> int:
    """Set the active scenario label on the victim EC2 instance."""
    return victim_set_scenario(
        project_root=PROJECT_ROOT,
        scenario=args.scenario,
    )


def deploy_victim_sender_command(_: argparse.Namespace) -> int:
    """Copy the victim PCAP sender script to the victim EC2 instance."""
    return deploy_victim_sender(PROJECT_ROOT)


def verify_victim_sender_command(_: argparse.Namespace) -> int:
    """Verify the victim PCAP sender script on the victim EC2 instance."""
    return verify_victim_sender(PROJECT_ROOT)


def victim_send_pcap_command(args: argparse.Namespace) -> int:
    """Send one PCAP file from victim to IDS."""
    return victim_send_pcap(
        project_root=PROJECT_ROOT,
        pcap_path=args.pcap_path,
        scenario=args.scenario,
    )


def victim_send_pending_pcaps_command(_: argparse.Namespace) -> int:
    """Send all pending PCAP files from victim to IDS."""
    return victim_send_pending_pcaps(PROJECT_ROOT)


def deploy_victim_sender_agent_command(_: argparse.Namespace) -> int:
    """Copy the victim pending PCAP sender agent to the victim EC2 instance."""
    return deploy_victim_sender_agent(PROJECT_ROOT)


def verify_victim_sender_agent_command(_: argparse.Namespace) -> int:
    """Verify the victim pending PCAP sender agent on the victim EC2 instance."""
    return verify_victim_sender_agent(PROJECT_ROOT)


def start_victim_sender_agent_command(_: argparse.Namespace) -> int:
    """Start the victim pending PCAP sender agent."""
    return start_victim_sender_agent(PROJECT_ROOT)


def stop_victim_sender_agent_command(_: argparse.Namespace) -> int:
    """Stop the victim pending PCAP sender agent."""
    return stop_victim_sender_agent(PROJECT_ROOT)


def victim_sender_agent_status_command(_: argparse.Namespace) -> int:
    """Show victim pending PCAP sender agent status."""
    return victim_sender_agent_status(PROJECT_ROOT)


def victim_start_agents_command(_: argparse.Namespace) -> int:
    """Start both victim background agents."""
    capture_status = start_victim_capture_agent(PROJECT_ROOT)
    if capture_status != 0:
        return capture_status

    sender_status = start_victim_sender_agent(PROJECT_ROOT)
    if sender_status != 0:
        return sender_status

    return 0


def victim_stop_agents_command(_: argparse.Namespace) -> int:
    """Stop both victim background agents."""
    sender_status = stop_victim_sender_agent(PROJECT_ROOT)
    if sender_status != 0:
        return sender_status

    capture_status = stop_victim_capture_agent(PROJECT_ROOT)
    if capture_status != 0:
        return capture_status

    return 0


def victim_agents_status_command(_: argparse.Namespace) -> int:
    """Show both victim background agent statuses."""
    capture_status = victim_capture_agent_status(PROJECT_ROOT)
    if capture_status != 0:
        return capture_status

    sender_status = victim_sender_agent_status(PROJECT_ROOT)
    if sender_status != 0:
        return sender_status

    return 0


def deploy_victim_capture_command(_: argparse.Namespace) -> int:
    """Copy the victim capture script to the victim EC2 instance."""
    return deploy_victim_capture(PROJECT_ROOT)


def verify_victim_capture_command(_: argparse.Namespace) -> int:
    """Verify the victim capture script on the victim EC2 instance."""
    return verify_victim_capture(PROJECT_ROOT)


def victim_capture_scenario_command(args: argparse.Namespace) -> int:
    """Capture one scenario on the victim EC2 instance."""
    return victim_capture_scenario(
        project_root=PROJECT_ROOT,
        scenario=args.scenario,
        seconds=args.seconds,
    )


def victim_list_pcaps_command(_: argparse.Namespace) -> int:
    """List PCAP files on the victim EC2 instance."""
    return victim_list_pcaps(PROJECT_ROOT)


def deploy_victim_capture_agent_command(_: argparse.Namespace) -> int:
    """Copy the victim continuous capture agent to the victim EC2 instance."""
    return deploy_victim_capture_agent(PROJECT_ROOT)


def verify_victim_capture_agent_command(_: argparse.Namespace) -> int:
    """Verify the victim continuous capture agent on the victim EC2 instance."""
    return verify_victim_capture_agent(PROJECT_ROOT)


def start_victim_capture_agent_command(_: argparse.Namespace) -> int:
    """Start the victim continuous capture agent."""
    return start_victim_capture_agent(PROJECT_ROOT)


def stop_victim_capture_agent_command(_: argparse.Namespace) -> int:
    """Stop the victim continuous capture agent."""
    return stop_victim_capture_agent(PROJECT_ROOT)


def victim_capture_agent_status_command(_: argparse.Namespace) -> int:
    """Show victim continuous capture agent status."""
    return victim_capture_agent_status(PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    """Create the command parser."""
    parser = argparse.ArgumentParser(description="AWS IDS Testbed 06 controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(handler=status)

    show_config_parser = subparsers.add_parser("show-config")
    show_config_parser.set_defaults(handler=show_config)

    # This command shows the saved EC2 information from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli show-inventory
    show_inventory_parser = subparsers.add_parser("show-inventory")
    show_inventory_parser.set_defaults(handler=show_inventory)

    # This command creates only the victim EC2 instance.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli create-victim
    create_victim_parser = subparsers.add_parser("create-victim")
    create_victim_parser.set_defaults(handler=create_victim)

    # This command creates only the attacker EC2 instance.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli create-attacker
    create_attacker_parser = subparsers.add_parser("create-attacker")
    create_attacker_parser.set_defaults(handler=create_attacker)

    # This command creates only the IDS EC2 instance.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli create-ids
    create_ids_parser = subparsers.add_parser("create-ids")
    create_ids_parser.set_defaults(handler=create_ids)

    # These commands refresh saved EC2 information from AWS.
    # They are useful after instances move from pending to running.
    refresh_victim_parser = subparsers.add_parser("refresh-victim")
    refresh_victim_parser.set_defaults(handler=refresh_victim)

    refresh_attacker_parser = subparsers.add_parser("refresh-attacker")
    refresh_attacker_parser.set_defaults(handler=refresh_attacker)

    refresh_ids_parser = subparsers.add_parser("refresh-ids")
    refresh_ids_parser.set_defaults(handler=refresh_ids)

    # These commands terminate one saved EC2 instance.
    # Use them when you want to stop AWS cost for that instance.
    terminate_victim_parser = subparsers.add_parser("terminate-victim")
    terminate_victim_parser.set_defaults(handler=terminate_victim)

    terminate_attacker_parser = subparsers.add_parser("terminate-attacker")
    terminate_attacker_parser.set_defaults(handler=terminate_attacker)

    terminate_ids_parser = subparsers.add_parser("terminate-ids")
    terminate_ids_parser.set_defaults(handler=terminate_ids)

    # This command sends scripts/setup_victim.sh to the victim EC2 instance.
    # Then it runs that script on the victim.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli setup-victim
    setup_victim_parser = subparsers.add_parser("setup-victim")
    setup_victim_parser.set_defaults(handler=setup_victim)

    # This command sends scripts/setup_attacker.sh to the attacker EC2 instance.
    # Then it runs that script on the attacker.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli setup-attacker
    setup_attacker_parser = subparsers.add_parser("setup-attacker")
    setup_attacker_parser.set_defaults(handler=setup_attacker)

    # This command sends scripts/setup_ids.sh to the IDS EC2 instance.
    # Then it runs that script on the IDS machine.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli setup-ids
    setup_ids_parser = subparsers.add_parser("setup-ids")
    setup_ids_parser.set_defaults(handler=setup_ids)

    # This command reads the victim public IP from inventory.yaml.
    # Then it runs tcpdump on the victim for 20 seconds.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli capture-benign-http
    capture_benign_http_parser = subparsers.add_parser("capture-benign-http")
    capture_benign_http_parser.set_defaults(handler=capture_benign_http)

    # This command reads victim public IP from inventory.yaml.
    # Then it checks the benign HTTP PCAP file on the victim.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli verify-benign-pcap
    verify_benign_pcap_parser = subparsers.add_parser("verify-benign-pcap")
    verify_benign_pcap_parser.set_defaults(handler=verify_benign_pcap)

    # This command reads attacker public IP and victim private IP from inventory.yaml.
    # Then it runs ApacheBench from attacker to victim over the AWS private network.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli generate-benign-http
    generate_benign_http_parser = subparsers.add_parser("generate-benign-http")
    generate_benign_http_parser.set_defaults(handler=generate_benign_http_traffic)

    # This command generates selected lab traffic from attacker to victim.
    # Traffic code:
    #   1 = benign HTTP
    #   2 = controlled DoS HTTP flood
    #   3 = controlled DoS SYN flood
    # Terminal command examples:
    #   python3 -m aws_ids_testbed_06.cli generate-traffic 1 --requests 100 --concurrency 5
    #   python3 -m aws_ids_testbed_06.cli generate-traffic 2 --requests 1000 --concurrency 50
    #   python3 -m aws_ids_testbed_06.cli generate-traffic 3 --packet-count 1000 --port 80
    generate_traffic_parser = subparsers.add_parser("generate-traffic")
    generate_traffic_parser.add_argument(
        "traffic_code",
        choices=["1", "2", "3"],
        help="1=benign_http, 2=dos_http_flood, 3=dos_syn_flood",
    )
    generate_traffic_parser.add_argument("--requests", type=int)
    generate_traffic_parser.add_argument("--concurrency", type=int)
    generate_traffic_parser.add_argument("--packet-count", type=int)
    generate_traffic_parser.add_argument("--port", type=int, default=80)
    generate_traffic_parser.set_defaults(handler=generate_traffic_command)

    # This command saves victim target settings on the attacker.
    # It reads the victim private IP from inventory.yaml.
    # Then it writes /opt/aws_ids_testbed/config/attacker.env on the attacker.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli configure-attacker-victim-url
    configure_attacker_victim_parser = subparsers.add_parser(
        "configure-attacker-victim-url"
    )
    configure_attacker_victim_parser.set_defaults(handler=configure_attacker_victim)

    # This command verifies victim target settings saved on the attacker.
    # It reads the attacker public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli attacker-verify-victim-url
    attacker_verify_victim_url_parser = subparsers.add_parser(
        "attacker-verify-victim-url"
    )
    attacker_verify_victim_url_parser.set_defaults(
        handler=attacker_verify_victim_url_command
    )

    # This command copies the attacker traffic script to the attacker EC2 instance.
    # It reads the attacker public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli deploy-attacker-traffic
    deploy_attacker_traffic_parser = subparsers.add_parser("deploy-attacker-traffic")
    deploy_attacker_traffic_parser.set_defaults(handler=deploy_attacker_traffic_command)

    # This command verifies the attacker traffic script on the attacker EC2 instance.
    # It reads the attacker public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli verify-attacker-traffic
    verify_attacker_traffic_parser = subparsers.add_parser("verify-attacker-traffic")
    verify_attacker_traffic_parser.set_defaults(handler=verify_attacker_traffic_command)

    # This command runs attacker-side traffic generation.
    # The attacker reads victim target settings from attacker.env.
    # Traffic code:
    #   1 = benign HTTP
    #   2 = controlled DoS HTTP flood
    #   3 = controlled DoS SYN flood
    # Terminal command examples:
    #   python3 -m aws_ids_testbed_06.cli attacker-generate-traffic 1
    #   python3 -m aws_ids_testbed_06.cli attacker-generate-traffic 2 --requests 1000 --concurrency 50
    #   python3 -m aws_ids_testbed_06.cli attacker-generate-traffic 3 --packet-count 100 --port 80
    attacker_generate_traffic_parser = subparsers.add_parser(
        "attacker-generate-traffic"
    )
    attacker_generate_traffic_parser.add_argument(
        "traffic_code",
        choices=["1", "2", "3"],
        help="1=benign_http, 2=dos_http_flood, 3=dos_syn_flood",
    )
    attacker_generate_traffic_parser.add_argument("--requests", type=int)
    attacker_generate_traffic_parser.add_argument("--concurrency", type=int)
    attacker_generate_traffic_parser.add_argument("--packet-count", type=int)
    attacker_generate_traffic_parser.add_argument("--port", type=int)
    attacker_generate_traffic_parser.set_defaults(
        handler=attacker_generate_traffic_command
    )

    # This command coordinates victim labeling/capture with attacker traffic generation.
    # It keeps the instances autonomous:
    #   victim reads ACTIVE_SCENARIO and captures/sends in the background
    #   attacker reads attacker.env and generates the selected traffic
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli run-scenario 1 --requests 100 --concurrency 5
    run_scenario_parser = subparsers.add_parser("run-scenario")
    run_scenario_parser.add_argument(
        "traffic_code",
        choices=["1", "2", "3"],
        help="1=benign_http, 2=dos_http_flood, 3=dos_syn_flood",
    )
    run_scenario_parser.add_argument("--requests", type=int)
    run_scenario_parser.add_argument("--concurrency", type=int)
    run_scenario_parser.add_argument("--packet-count", type=int)
    run_scenario_parser.add_argument("--port", type=int)
    run_scenario_parser.add_argument(
        "--auto-stop-seconds",
        type=int,
        help="wait this many seconds after traffic generation, then stop victim agents",
    )
    run_scenario_parser.set_defaults(handler=run_scenario_command)

    # This command copies only essential receiver files to the IDS EC2 instance.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli deploy-ids-receiver
    deploy_ids_receiver_parser = subparsers.add_parser("deploy-ids-receiver")
    deploy_ids_receiver_parser.set_defaults(handler=deploy_ids_receiver)

    # This command starts the IDS FastAPI receiver in the background.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli ids-start-receiver
    ids_start_receiver_parser = subparsers.add_parser("ids-start-receiver")
    ids_start_receiver_parser.set_defaults(handler=ids_start_receiver_command)

    # This command lists PCAP files received by the IDS receiver.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli ids-list-received-pcaps
    ids_list_received_pcaps_parser = subparsers.add_parser("ids-list-received-pcaps")
    ids_list_received_pcaps_parser.set_defaults(handler=ids_list_received_pcaps_command)

    # This command saves IDS receiver settings on the victim.
    # It reads the IDS private IP from inventory.yaml.
    # Then it writes /opt/aws_ids_testbed/config/victim.env on the victim.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli configure-victim-ids-url
    configure_victim_ids_parser = subparsers.add_parser("configure-victim-ids-url")
    configure_victim_ids_parser.set_defaults(handler=configure_victim_ids)

    # This command verifies IDS receiver settings saved on the victim.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-verify-ids-url
    victim_verify_ids_url_parser = subparsers.add_parser("victim-verify-ids-url")
    victim_verify_ids_url_parser.set_defaults(handler=victim_verify_ids_url_command)

    # This command updates ACTIVE_SCENARIO in victim.env.
    # The future continuous capture agent will use this label when naming PCAP chunks.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-set-scenario benign_http
    victim_set_scenario_parser = subparsers.add_parser("victim-set-scenario")
    victim_set_scenario_parser.add_argument(
        "scenario",
        choices=["benign_http", "dos_http_flood", "dos_syn_flood"],
    )
    victim_set_scenario_parser.set_defaults(handler=victim_set_scenario_command)

    # This command copies the victim PCAP sender script to the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli deploy-victim-sender
    deploy_victim_sender_parser = subparsers.add_parser("deploy-victim-sender")
    deploy_victim_sender_parser.set_defaults(handler=deploy_victim_sender_command)

    # This command verifies the victim PCAP sender script on the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli verify-victim-sender
    verify_victim_sender_parser = subparsers.add_parser("verify-victim-sender")
    verify_victim_sender_parser.set_defaults(handler=verify_victim_sender_command)

    # This command sends any completed PCAP file from victim to IDS.
    # It reads the victim public IP from inventory.yaml.
    # The victim sender script reads IDS_RECEIVER_URL from victim.env.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-send-pcap --pcap-path PATH --scenario LABEL
    victim_send_pcap_parser = subparsers.add_parser("victim-send-pcap")
    victim_send_pcap_parser.add_argument("--pcap-path", required=True)
    victim_send_pcap_parser.add_argument("--scenario", default="unknown")
    victim_send_pcap_parser.set_defaults(handler=victim_send_pcap_command)

    # This command sends all completed pending PCAP files from victim to IDS.
    # It does not require a hard-coded PCAP filename or scenario.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-send-pending-pcaps
    victim_send_pending_pcaps_parser = subparsers.add_parser(
        "victim-send-pending-pcaps"
    )
    victim_send_pending_pcaps_parser.set_defaults(
        handler=victim_send_pending_pcaps_command
    )

    # This command copies the pending PCAP sender agent to the victim EC2 instance.
    # The agent will later upload pending PCAP files to IDS in the background.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli deploy-victim-sender-agent
    deploy_victim_sender_agent_parser = subparsers.add_parser(
        "deploy-victim-sender-agent"
    )
    deploy_victim_sender_agent_parser.set_defaults(
        handler=deploy_victim_sender_agent_command
    )

    # This command verifies the pending PCAP sender agent on the victim EC2 instance.
    # It checks that the script exists, is executable, and has valid bash syntax.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli verify-victim-sender-agent
    verify_victim_sender_agent_parser = subparsers.add_parser(
        "verify-victim-sender-agent"
    )
    verify_victim_sender_agent_parser.set_defaults(
        handler=verify_victim_sender_agent_command
    )

    # This command starts the pending PCAP sender agent in the background.
    # It writes logs to /opt/aws_ids_testbed/logs/pending_pcap_sender_agent.log.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-start-sender-agent
    victim_start_sender_agent_parser = subparsers.add_parser(
        "victim-start-sender-agent"
    )
    victim_start_sender_agent_parser.set_defaults(
        handler=start_victim_sender_agent_command
    )

    # This command stops the pending PCAP sender agent.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-stop-sender-agent
    victim_stop_sender_agent_parser = subparsers.add_parser(
        "victim-stop-sender-agent"
    )
    victim_stop_sender_agent_parser.set_defaults(
        handler=stop_victim_sender_agent_command
    )

    # This command shows whether the pending PCAP sender agent is running.
    # It also prints recent sender-agent log lines.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-sender-agent-status
    victim_sender_agent_status_parser = subparsers.add_parser(
        "victim-sender-agent-status"
    )
    victim_sender_agent_status_parser.set_defaults(
        handler=victim_sender_agent_status_command
    )

    # This command starts both victim background agents.
    # It starts:
    #   1. continuous capture agent
    #   2. pending PCAP sender agent
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-start-agents
    victim_start_agents_parser = subparsers.add_parser("victim-start-agents")
    victim_start_agents_parser.set_defaults(handler=victim_start_agents_command)

    # This command stops both victim background agents.
    # It stops:
    #   1. pending PCAP sender agent
    #   2. continuous capture agent
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-stop-agents
    victim_stop_agents_parser = subparsers.add_parser("victim-stop-agents")
    victim_stop_agents_parser.set_defaults(handler=victim_stop_agents_command)

    # This command shows both victim background agent statuses.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-agents-status
    victim_agents_status_parser = subparsers.add_parser("victim-agents-status")
    victim_agents_status_parser.set_defaults(handler=victim_agents_status_command)

    # This command copies the victim scenario capture script to the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli deploy-victim-capture
    deploy_victim_capture_parser = subparsers.add_parser("deploy-victim-capture")
    deploy_victim_capture_parser.set_defaults(handler=deploy_victim_capture_command)

    # This command verifies the victim scenario capture script on the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli verify-victim-capture
    verify_victim_capture_parser = subparsers.add_parser("verify-victim-capture")
    verify_victim_capture_parser.set_defaults(handler=verify_victim_capture_command)

    # This command captures one scenario on the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-capture-scenario --scenario LABEL --seconds 20
    victim_capture_scenario_parser = subparsers.add_parser("victim-capture-scenario")
    victim_capture_scenario_parser.add_argument("--scenario", required=True)
    victim_capture_scenario_parser.add_argument("--seconds", type=int, default=20)
    victim_capture_scenario_parser.set_defaults(handler=victim_capture_scenario_command)

    # This command lists victim PCAP files in writing, pending, sent, and failed.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-list-pcaps
    victim_list_pcaps_parser = subparsers.add_parser("victim-list-pcaps")
    victim_list_pcaps_parser.set_defaults(handler=victim_list_pcaps_command)

    # This command copies the continuous capture agent to the victim EC2 instance.
    # The agent will later capture 10-second PCAP chunks in the background.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli deploy-victim-capture-agent
    deploy_victim_capture_agent_parser = subparsers.add_parser(
        "deploy-victim-capture-agent"
    )
    deploy_victim_capture_agent_parser.set_defaults(
        handler=deploy_victim_capture_agent_command
    )

    # This command verifies the continuous capture agent on the victim EC2 instance.
    # It checks that the script exists, is executable, and has valid bash syntax.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli verify-victim-capture-agent
    verify_victim_capture_agent_parser = subparsers.add_parser(
        "verify-victim-capture-agent"
    )
    verify_victim_capture_agent_parser.set_defaults(
        handler=verify_victim_capture_agent_command
    )

    # This command starts the continuous capture agent in the background.
    # It writes logs to /opt/aws_ids_testbed/logs/continuous_capture_agent.log.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-start-capture-agent
    victim_start_capture_agent_parser = subparsers.add_parser(
        "victim-start-capture-agent"
    )
    victim_start_capture_agent_parser.set_defaults(
        handler=start_victim_capture_agent_command
    )

    # This command stops the continuous capture agent.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-stop-capture-agent
    victim_stop_capture_agent_parser = subparsers.add_parser(
        "victim-stop-capture-agent"
    )
    victim_stop_capture_agent_parser.set_defaults(
        handler=stop_victim_capture_agent_command
    )

    # This command shows whether the continuous capture agent is running.
    # It also prints recent agent log lines.
    # Terminal command:
    #   python3 -m aws_ids_testbed_06.cli victim-capture-agent-status
    victim_capture_agent_status_parser = subparsers.add_parser(
        "victim-capture-agent-status"
    )
    victim_capture_agent_status_parser.set_defaults(
        handler=victim_capture_agent_status_command
    )

    return parser


def main() -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
