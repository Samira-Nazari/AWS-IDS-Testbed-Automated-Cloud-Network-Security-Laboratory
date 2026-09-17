"""Simple command line interface for AWS IDS Testbed 10."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ids_detector.lite_v6.config.config import (
    IDS_CAPTURE_MANIFEST_PATH,
    IDS_CSV_INPUT_DIR,
    IDS_PREDICTION_BATCH_SIZE,
    IDS_PREDICTION_DEVICE,
    IDS_PREDICTION_SUMMARY_PATH,
    IDS_PREDICTIONS_PATH,
    STEP_1_LOADED_DATA_PATH,
    STEP_2_PROCESSED_DATA_PATH,
    STEP_3_FEATURES_USED_PATH,
    STEP_3_WINDOW_DATA_PATH,
    STEP_3_WINDOW_METADATA_PATH,
    STEP_3_WINDOWING_SUMMARY_PATH,
)
from ids_detector.lite_v6.data.data_loader import load_data as load_ids_step_1_data
from ids_detector.lite_v6.evaluation.evaluator import (
    debug_ids_compare_lite_v6_classes,
    debug_ids_feature_distribution,
    debug_ids_inference_artifacts,
    debug_ids_prediction_probabilities,
    debug_ids_selected_feature_values,
    run_step_4,
)
from ids_detector.lite_v6.features.feature_engineering import run_step_3
from ids_detector.lite_v6.preprocessing.preprocessor import run_step_2

from aws_ids_testbed_10.attacker_config_service import (
    attacker_verify_victim_url,
    configure_attacker_victim_url,
)
from aws_ids_testbed_10.attacker_traffic_service import (
    attacker_generate_traffic,
    deploy_attacker_traffic,
    verify_attacker_traffic,
)
from aws_ids_testbed_10.benign_network_config_service import (
    configure_benign_network,
)
from aws_ids_testbed_10.benign_generator_lifecycle import (
    create_benign_generators,
    purge_benign_generators,
    refresh_benign_generators,
    terminate_benign_generators,
)
from aws_ids_testbed_10.benign_connectivity_service import (
    verify_benign_connectivity,
)
from aws_ids_testbed_10.benign_traffic_service import (
    benign_traffic_status,
    deploy_benign_traffic,
    diagnose_benign_generators,
    start_benign_traffic,
    stop_benign_traffic,
    verify_benign_traffic,
)
from aws_ids_testbed_10.capture_service import (
    capture_benign_http_on_victim,
    verify_benign_http_pcap,
)
from aws_ids_testbed_10.config import load_config
from aws_ids_testbed_10.ec2_lab import (
    create_one_instance,
    refresh_instance_info,
    terminate_instance,
)
from aws_ids_testbed_10.ids_receiver_service import (
    deploy_ids_receiver_files,
    ids_list_received_pcaps,
    ids_start_receiver,
)
from aws_ids_testbed_10.ids_csv_download_service import download_ids_csv_files
from aws_ids_testbed_10.ids_pcap_converter_service import (
    deploy_ids_pcap_converter,
    ids_convert_pcap,
    ids_convert_received_pcaps,
    ids_delete_pcap_csv,
    ids_list_csv_files,
    ids_pcap_converter_agent_status,
    start_ids_pcap_converter_agent,
    stop_ids_pcap_converter_agent,
    verify_ids_pcap_converter,
)
from aws_ids_testbed_10.inventory import load_inventory, update_instance
from aws_ids_testbed_10.setup_service import (
    run_setup_for_benign_generators,
    run_setup_for_role,
)
from aws_ids_testbed_10.traffic_service import (
    generate_benign_http,
    generate_traffic_by_code,
)
from aws_ids_testbed_10.victim_capture_service import (
    deploy_victim_capture,
    verify_benign_pcap_protocols,
    verify_victim_capture,
    victim_capture_scenario,
    victim_delete_pcap_csv,
    victim_list_pcaps,
)
from aws_ids_testbed_10.victim_capture_agent_service import (
    deploy_victim_capture_agent,
    start_victim_capture_agent,
    stop_victim_capture_agent,
    victim_capture_agent_status,
    verify_victim_capture_agent,
)
from aws_ids_testbed_10.victim_benign_service import (
    setup_victim_benign_services,
    verify_victim_benign_services,
)
from aws_ids_testbed_10.victim_config_service import (
    configure_victim_ids_url,
    victim_set_scenario,
    victim_verify_ids_url,
)
from aws_ids_testbed_10.victim_sender_service import (
    deploy_victim_sender,
    verify_victim_sender,
    victim_send_pcap,
    victim_send_pending_pcaps,
)
from aws_ids_testbed_10.victim_sender_agent_service import (
    deploy_victim_sender_agent,
    start_victim_sender_agent,
    stop_victim_sender_agent,
    victim_sender_agent_status,
    verify_victim_sender_agent,
    verify_victim_sender_delivery,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENDER_DRAIN_SECONDS = 10


def status(_: argparse.Namespace) -> int:
    """Show basic project status."""
    print("AWS IDS Testbed 10")
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


def create_benign_generators_command(_: argparse.Namespace) -> int:
    """Create the five benign generator EC2 instances."""
    return create_benign_generators(PROJECT_ROOT)


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


def refresh_benign_generators_command(_: argparse.Namespace) -> int:
    """Refresh the five benign generator EC2 instances."""
    return refresh_benign_generators(PROJECT_ROOT)


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


def terminate_benign_generators_command(_: argparse.Namespace) -> int:
    """Terminate the five benign generator EC2 instances."""
    return terminate_benign_generators(PROJECT_ROOT)


def purge_benign_generators_command(args: argparse.Namespace) -> int:
    """Finish bounded cleanup and remove stale benign-generator inventory."""
    if not args.confirm:
        print("Refusing cleanup without --confirm.")
        print("Re-run with: purge-benign-generators --confirm")
        return 2
    return purge_benign_generators(PROJECT_ROOT)


def setup_victim(_: argparse.Namespace) -> int:
    """Run setup_victim.sh on the victim EC2 instance."""
    return run_setup_for_role(PROJECT_ROOT, "victim")


def setup_attacker(_: argparse.Namespace) -> int:
    """Run setup_attacker.sh on the attacker EC2 instance."""
    return run_setup_for_role(PROJECT_ROOT, "attacker")


def setup_ids(_: argparse.Namespace) -> int:
    """Run setup_ids.sh on the IDS EC2 instance."""
    return run_setup_for_role(PROJECT_ROOT, "ids")


def setup_benign_generators(_: argparse.Namespace) -> int:
    """Install benign traffic tools on all five generators."""
    return run_setup_for_benign_generators(PROJECT_ROOT)


def setup_victim_benign_services_command(_: argparse.Namespace) -> int:
    """Configure victim-side benign IoT-like services."""
    return setup_victim_benign_services(PROJECT_ROOT)


def verify_victim_benign_services_command(
    _: argparse.Namespace,
) -> int:
    """Verify victim benign services and listening ports."""
    return verify_victim_benign_services(PROJECT_ROOT)


def configure_benign_network_command(_: argparse.Namespace) -> int:
    """Save current victim/generator private IP settings."""
    return configure_benign_network(PROJECT_ROOT)


def verify_benign_connectivity_command(
    _: argparse.Namespace,
) -> int:
    """Verify benign connectivity from all generators."""
    return verify_benign_connectivity(PROJECT_ROOT)


def deploy_benign_traffic_command(_: argparse.Namespace) -> int:
    """Deploy the benign traffic agent to all generators."""
    return deploy_benign_traffic(PROJECT_ROOT)


def verify_benign_traffic_command(_: argparse.Namespace) -> int:
    """Verify the benign traffic agent on all generators."""
    return verify_benign_traffic(PROJECT_ROOT)


def start_benign_traffic_command(args: argparse.Namespace) -> int:
    """Start all benign generators for a fixed duration."""
    return start_benign_traffic(
        PROJECT_ROOT,
        duration_seconds=args.duration_seconds,
    )


def stop_benign_traffic_command(_: argparse.Namespace) -> int:
    """Stop all running benign generators."""
    return stop_benign_traffic(PROJECT_ROOT)


def benign_traffic_status_command(_: argparse.Namespace) -> int:
    """Show the status of all benign generators."""
    return benign_traffic_status(PROJECT_ROOT)


def diagnose_benign_generators_command(args: argparse.Namespace) -> int:
    """Run foreground diagnostics on all benign generators."""
    return diagnose_benign_generators(
        project_root=PROJECT_ROOT,
        duration_seconds=args.duration_seconds,
    )


def run_benign_scenario_command(args: argparse.Namespace) -> int:
    """Run five benign generators with victim capture and sender agents."""
    duration_seconds = args.auto_stop_seconds

    if duration_seconds is None or duration_seconds <= 0:
        print(
            "The benign scenario requires "
            "--auto-stop-seconds with a positive value."
        )
        return 1

    print("Scenario: benign_iot_5_generators")

    print("Step 1: Set victim ACTIVE_SCENARIO.")
    scenario_status = victim_set_scenario(
        project_root=PROJECT_ROOT,
        scenario="benign_http",
    )
    if scenario_status != 0:
        return scenario_status

    print("Step 2: Start victim capture and sender agents.")
    agents_status = victim_start_agents_command(args)
    if agents_status != 0:
        victim_stop_agents_command(args)
        return agents_status

    print("Step 3: Start all five benign generators.")
    traffic_status = start_benign_traffic(
        project_root=PROJECT_ROOT,
        duration_seconds=duration_seconds,
    )
    if traffic_status != 0:
        stop_benign_traffic(PROJECT_ROOT)
        victim_stop_agents_command(args)
        return traffic_status

    try:
        print(
            f"Step 4: Generate benign traffic for "
            f"{duration_seconds} seconds."
        )
        time.sleep(duration_seconds)
    finally:
        print("Step 5: Stop all five benign generators.")
        traffic_stop_status = stop_benign_traffic(PROJECT_ROOT)

        print("Step 6: Stop victim capture.")
        capture_stop_status = stop_victim_capture_agent(PROJECT_ROOT)

        print(
            "Step 7: Drain pending PCAP uploads for "
            f"{SENDER_DRAIN_SECONDS} seconds."
        )
        time.sleep(SENDER_DRAIN_SECONDS)

        print("Step 8: Stop victim sender agent.")
        sender_stop_status = stop_victim_sender_agent(PROJECT_ROOT)

        print("Step 9: Verify PCAP delivery.")
        delivery_status = verify_victim_sender_delivery(
            PROJECT_ROOT,
            scenario="benign_http",
        )

    if traffic_stop_status != 0:
        return traffic_stop_status

    if capture_stop_status != 0:
        return capture_stop_status

    if sender_stop_status != 0:
        return sender_stop_status

    return delivery_status


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
        interval_microseconds=args.interval_microseconds,
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
        interval_microseconds=args.interval_microseconds,
    )


def run_scenario_command(args: argparse.Namespace) -> int:
    """Run one labeled scenario with coordinated victim and attacker actions."""
    if args.traffic_code == "4":
        return run_benign_scenario_command(args)

    if args.auto_stop_seconds is None or args.auto_stop_seconds <= 0:
        print(
            "Scenarios 1, 2, and 3 require "
            "--auto-stop-seconds with a positive value."
        )
        return 1

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
        victim_stop_agents_command(args)
        return agents_status

    traffic_status = 0

    try:
        print("Step 3: Generate matching traffic from attacker.")
        traffic_status = attacker_generate_traffic(
            project_root=PROJECT_ROOT,
            traffic_code=args.traffic_code,
            requests=args.requests,
            concurrency=args.concurrency,
            packet_count=args.packet_count,
            port=args.port,
            interval_microseconds=args.interval_microseconds,
        )

        if traffic_status == 0:
            print(
                f"Step 4: Wait {args.auto_stop_seconds} seconds "
                "before stopping capture."
            )
            time.sleep(args.auto_stop_seconds)
    finally:
        print("Step 5: Stop victim capture.")
        capture_stop_status = stop_victim_capture_agent(PROJECT_ROOT)

        print(
            "Step 6: Drain pending PCAP uploads for "
            f"{SENDER_DRAIN_SECONDS} seconds."
        )
        time.sleep(SENDER_DRAIN_SECONDS)

        print("Step 7: Stop victim sender agent.")
        sender_stop_status = stop_victim_sender_agent(PROJECT_ROOT)

        print("Step 8: Verify PCAP delivery.")
        delivery_status = verify_victim_sender_delivery(
            PROJECT_ROOT,
            scenario=scenario,
        )

    if traffic_status != 0:
        return traffic_status

    if capture_stop_status != 0:
        return capture_stop_status

    if sender_stop_status != 0:
        return sender_stop_status

    return delivery_status


def deploy_ids_receiver(_: argparse.Namespace) -> int:
    """Copy essential IDS receiver files to the IDS EC2 instance."""
    return deploy_ids_receiver_files(PROJECT_ROOT)


def ids_start_receiver_command(_: argparse.Namespace) -> int:
    """Start the IDS FastAPI receiver on the IDS EC2 instance."""
    return ids_start_receiver(PROJECT_ROOT)


def ids_list_received_pcaps_command(_: argparse.Namespace) -> int:
    """List PCAP files received on the IDS EC2 instance."""
    return ids_list_received_pcaps(PROJECT_ROOT)


def deploy_ids_pcap_converter_command(_: argparse.Namespace) -> int:
    """Deploy IDS PCAP-to-CSV converter files to the IDS EC2 instance."""
    return deploy_ids_pcap_converter(PROJECT_ROOT)


def verify_ids_pcap_converter_command(_: argparse.Namespace) -> int:
    """Verify IDS PCAP-to-CSV converter files on the IDS EC2 instance."""
    return verify_ids_pcap_converter(PROJECT_ROOT)


def start_ids_pcap_converter_agent_command(_: argparse.Namespace) -> int:
    """Start IDS PCAP-to-CSV converter agent on the IDS EC2 instance."""
    return start_ids_pcap_converter_agent(PROJECT_ROOT)


def stop_ids_pcap_converter_agent_command(_: argparse.Namespace) -> int:
    """Stop IDS PCAP-to-CSV converter agent on the IDS EC2 instance."""
    return stop_ids_pcap_converter_agent(PROJECT_ROOT)


def ids_pcap_converter_agent_status_command(_: argparse.Namespace) -> int:
    """Show IDS PCAP-to-CSV converter agent status on the IDS EC2 instance."""
    return ids_pcap_converter_agent_status(PROJECT_ROOT)


def ids_delete_pcap_csv_command(args: argparse.Namespace) -> int:
    """Delete runtime PCAP and CSV files from the IDS instance."""
    return ids_delete_pcap_csv(
        project_root=PROJECT_ROOT,
        confirm=args.confirm,
    )


def ids_list_csv_files_command(_: argparse.Namespace) -> int:
    """List converted CSV files on the IDS EC2 instance."""
    return ids_list_csv_files(PROJECT_ROOT)


def ids_download_csv_files_command(args: argparse.Namespace) -> int:
    """Download converted IDS CSV files to local Slurm."""
    return download_ids_csv_files(
        project_root=PROJECT_ROOT,
        csv_path=args.csv_path,
        local_output_directory=args.local_output_directory,
    )


def ids_run_inference_step1_command(args: argparse.Namespace) -> int:
    """Run local IDS Lite V6 inference STEP=1 on downloaded CSV files."""
    input_dir = Path(args.input_directory) if args.input_directory else IDS_CSV_INPUT_DIR
    output_path = Path(args.output_path) if args.output_path else STEP_1_LOADED_DATA_PATH
    manifest_path = (
        Path(args.manifest_path) if args.manifest_path else IDS_CAPTURE_MANIFEST_PATH
    )

    data = load_ids_step_1_data(
        input_dir=input_dir,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    print("[ids-inference-step1] Completed.")
    print(f"[ids-inference-step1] Input CSV directory: {input_dir}")
    print(f"[ids-inference-step1] Loaded data: {output_path}")
    print(f"[ids-inference-step1] Capture manifest: {manifest_path}")
    print(f"[ids-inference-step1] Rows: {len(data)}")
    print(f"[ids-inference-step1] Columns: {len(data.columns)}")
    return 0


def ids_run_inference_step2_command(args: argparse.Namespace) -> int:
    """Run local IDS Lite V6 inference STEP=2 preprocessing."""
    input_path = Path(args.input_path) if args.input_path else STEP_1_LOADED_DATA_PATH
    output_path = (
        Path(args.output_path) if args.output_path else STEP_2_PROCESSED_DATA_PATH
    )

    data = run_step_2(
        input_path=input_path,
        output_path=output_path,
    )

    print("[ids-inference-step2] Completed.")
    print(f"[ids-inference-step2] Input data: {input_path}")
    print(f"[ids-inference-step2] Processed data: {output_path}")
    print(f"[ids-inference-step2] Rows: {len(data)}")
    print(f"[ids-inference-step2] Columns: {len(data.columns)}")
    return 0


def ids_run_inference_step3_command(args: argparse.Namespace) -> int:
    """Run local IDS Lite V6 inference STEP=3 feature/window engineering."""
    input_path = Path(args.input_path) if args.input_path else STEP_2_PROCESSED_DATA_PATH
    window_output_path = (
        Path(args.window_output_path)
        if args.window_output_path
        else STEP_3_WINDOW_DATA_PATH
    )
    metadata_output_path = (
        Path(args.metadata_output_path)
        if args.metadata_output_path
        else STEP_3_WINDOW_METADATA_PATH
    )
    features_output_path = (
        Path(args.features_output_path)
        if args.features_output_path
        else STEP_3_FEATURES_USED_PATH
    )
    summary_output_path = (
        Path(args.summary_output_path)
        if args.summary_output_path
        else STEP_3_WINDOWING_SUMMARY_PATH
    )

    result = run_step_3(
        input_path=input_path,
        window_output_path=window_output_path,
        metadata_output_path=metadata_output_path,
        features_output_path=features_output_path,
        summary_output_path=summary_output_path,
    )

    print("[ids-inference-step3] Completed.")
    print(f"[ids-inference-step3] Input data: {input_path}")
    print(f"[ids-inference-step3] Windows: {window_output_path}")
    print(f"[ids-inference-step3] Window metadata: {metadata_output_path}")
    print(f"[ids-inference-step3] Features used: {features_output_path}")
    print(f"[ids-inference-step3] Summary: {summary_output_path}")
    print(f"[ids-inference-step3] Window shape: {result['X_windows'].shape}")
    print(f"[ids-inference-step3] Metadata rows: {len(result['window_metadata'])}")
    return 0


def ids_run_inference_step4_command(args: argparse.Namespace) -> int:
    """Run local IDS Lite V6 inference STEP=4 model prediction."""
    window_data_path = (
        Path(args.window_data_path)
        if args.window_data_path
        else STEP_3_WINDOW_DATA_PATH
    )
    metadata_path = (
        Path(args.metadata_path)
        if args.metadata_path
        else STEP_3_WINDOW_METADATA_PATH
    )
    predictions_output_path = (
        Path(args.predictions_output_path)
        if args.predictions_output_path
        else IDS_PREDICTIONS_PATH
    )
    summary_output_path = (
        Path(args.summary_output_path)
        if args.summary_output_path
        else IDS_PREDICTION_SUMMARY_PATH
    )
    batch_size = args.batch_size or IDS_PREDICTION_BATCH_SIZE
    device_setting = args.device or IDS_PREDICTION_DEVICE

    result = run_step_4(
        window_data_path=window_data_path,
        metadata_path=metadata_path,
        predictions_output_path=predictions_output_path,
        summary_output_path=summary_output_path,
        batch_size=batch_size,
        device_setting=device_setting,
    )
    summary = result["summary"]

    print("[ids-inference-step4] Completed.")
    print(f"[ids-inference-step4] Windows: {window_data_path}")
    print(f"[ids-inference-step4] Window metadata: {metadata_path}")
    print(f"[ids-inference-step4] Predictions: {predictions_output_path}")
    print(f"[ids-inference-step4] Summary: {summary_output_path}")
    print(f"[ids-inference-step4] Device: {summary['device']}")
    print(f"[ids-inference-step4] Batch size: {summary['batch_size']}")
    print(f"[ids-inference-step4] Prediction rows: {summary['prediction_rows']}")
    print(f"[ids-inference-step4] Source files: {summary['source_files']}")
    print(f"[ids-inference-step4] Prediction counts: {summary['prediction_counts']}")
    if summary["expected_accuracy"] is not None:
        print(
            "[ids-inference-step4] Expected-label agreement: "
            f"{summary['expected_accuracy']:.4f}"
        )
    return 0


def ids_debug_inference_artifacts_command(args: argparse.Namespace) -> int:
    """Print read-only debug report for IDS Lite V6 inference artifacts."""
    report = debug_ids_inference_artifacts(
        step_2_data_path=(
            Path(args.step_2_data_path)
            if args.step_2_data_path
            else STEP_2_PROCESSED_DATA_PATH
        ),
        window_data_path=(
            Path(args.window_data_path)
            if args.window_data_path
            else STEP_3_WINDOW_DATA_PATH
        ),
        metadata_path=(
            Path(args.metadata_path)
            if args.metadata_path
            else STEP_3_WINDOW_METADATA_PATH
        ),
        features_used_path=(
            Path(args.features_used_path)
            if args.features_used_path
            else STEP_3_FEATURES_USED_PATH
        ),
    )
    print(yaml.safe_dump(report, sort_keys=False).rstrip())
    return 0


def ids_debug_feature_distribution_command(args: argparse.Namespace) -> int:
    """Print read-only debug report for IDS feature distribution."""
    report = debug_ids_feature_distribution(
        step_2_data_path=(
            Path(args.step_2_data_path)
            if args.step_2_data_path
            else STEP_2_PROCESSED_DATA_PATH
        ),
        top_n=args.top_n,
    )
    print(yaml.safe_dump(report, sort_keys=False).rstrip())
    return 0


def ids_debug_selected_feature_values_command(args: argparse.Namespace) -> int:
    """Print raw and scaled values for selected IDS features."""
    report = debug_ids_selected_feature_values(
        step_2_data_path=(
            Path(args.step_2_data_path)
            if args.step_2_data_path
            else STEP_2_PROCESSED_DATA_PATH
        ),
        features=args.features,
    )
    print(yaml.safe_dump(report, sort_keys=False).rstrip())
    return 0


def ids_debug_prediction_probabilities_command(args: argparse.Namespace) -> int:
    """Print read-only top-k probability report for IDS predictions."""
    report = debug_ids_prediction_probabilities(
        predictions_path=(
            Path(args.predictions_path)
            if args.predictions_path
            else IDS_PREDICTIONS_PATH
        ),
        top_k=args.top_k,
        sample_windows_per_file=args.sample_windows_per_file,
    )
    print(yaml.safe_dump(report, sort_keys=False).rstrip())
    return 0


def ids_debug_compare_lite_v6_classes_command(args: argparse.Namespace) -> int:
    """Compare AWS IDS windows against Lite V6 saved class windows."""
    report = debug_ids_compare_lite_v6_classes(
        window_data_path=(
            Path(args.window_data_path)
            if args.window_data_path
            else STEP_3_WINDOW_DATA_PATH
        ),
        metadata_path=(
            Path(args.metadata_path)
            if args.metadata_path
            else STEP_3_WINDOW_METADATA_PATH
        ),
        labels=args.labels,
        sample_windows_per_label=args.sample_windows_per_label,
        top_n_features=args.top_n_features,
        source_file_prefixes=args.source_file_prefixes,
    )
    print(yaml.safe_dump(report, sort_keys=False).rstrip())
    return 0


def ids_convert_pcap_command(args: argparse.Namespace) -> int:
    """Convert one received PCAP file to CSV on IDS."""
    return ids_convert_pcap(
        project_root=PROJECT_ROOT,
        pcap_path=args.pcap_path,
    )


def ids_convert_received_pcaps_command(_: argparse.Namespace) -> int:
    """Convert all received PCAP files to CSV on IDS."""
    return ids_convert_received_pcaps(PROJECT_ROOT)


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


def verify_victim_sender_delivery_command(_: argparse.Namespace) -> int:
    """Verify that victim PCAP files were delivered."""
    return verify_victim_sender_delivery(PROJECT_ROOT)


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


def victim_delete_pcap_csv_command(args: argparse.Namespace) -> int:
    """Delete runtime PCAP and CSV files from the victim."""
    return victim_delete_pcap_csv(
        project_root=PROJECT_ROOT,
        confirm=args.confirm,
    )


def verify_benign_pcap_protocols_command(_: argparse.Namespace) -> int:
    """Verify expected benign protocols in the newest pending victim PCAP."""
    return verify_benign_pcap_protocols(PROJECT_ROOT)


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
    parser = argparse.ArgumentParser(description="AWS IDS Testbed 10 controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(handler=status)

    show_config_parser = subparsers.add_parser("show-config")
    show_config_parser.set_defaults(handler=show_config)

    # This command shows the saved EC2 information from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli show-inventory
    show_inventory_parser = subparsers.add_parser("show-inventory")
    show_inventory_parser.set_defaults(handler=show_inventory)

    # This command creates only the victim EC2 instance.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli create-victim
    create_victim_parser = subparsers.add_parser("create-victim")
    create_victim_parser.set_defaults(handler=create_victim)

    # This command creates only the attacker EC2 instance.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli create-attacker
    create_attacker_parser = subparsers.add_parser("create-attacker")
    create_attacker_parser.set_defaults(handler=create_attacker)

    # This command creates only the IDS EC2 instance.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli create-ids
    create_ids_parser = subparsers.add_parser("create-ids")
    create_ids_parser.set_defaults(handler=create_ids)

    create_benign_generators_parser = subparsers.add_parser(
        "create-benign-generators",
        help="Create five benign traffic generator instances",
    )
    create_benign_generators_parser.set_defaults(
        handler=create_benign_generators_command
    )

    # These commands refresh saved EC2 information from AWS.
    # They are useful after instances move from pending to running.
    refresh_victim_parser = subparsers.add_parser("refresh-victim")
    refresh_victim_parser.set_defaults(handler=refresh_victim)

    refresh_attacker_parser = subparsers.add_parser("refresh-attacker")
    refresh_attacker_parser.set_defaults(handler=refresh_attacker)

    refresh_ids_parser = subparsers.add_parser("refresh-ids")
    refresh_ids_parser.set_defaults(handler=refresh_ids)

    refresh_benign_generators_parser = subparsers.add_parser(
        "refresh-benign-generators",
        help="Refresh five benign traffic generator instances",
    )
    refresh_benign_generators_parser.set_defaults(
        handler=refresh_benign_generators_command
    )

    # These commands terminate one saved EC2 instance.
    # Use them when you want to stop AWS cost for that instance.
    terminate_victim_parser = subparsers.add_parser("terminate-victim")
    terminate_victim_parser.set_defaults(handler=terminate_victim)

    terminate_attacker_parser = subparsers.add_parser("terminate-attacker")
    terminate_attacker_parser.set_defaults(handler=terminate_attacker)

    terminate_ids_parser = subparsers.add_parser("terminate-ids")
    terminate_ids_parser.set_defaults(handler=terminate_ids)

    terminate_benign_generators_parser = subparsers.add_parser(
        "terminate-benign-generators",
        help="Terminate five benign traffic generator instances",
    )
    terminate_benign_generators_parser.set_defaults(
        handler=terminate_benign_generators_command
    )

    purge_benign_generators_parser = subparsers.add_parser(
        "purge-benign-generators",
        help="Finish cleanup and remove stale benign-generator inventory entries",
    )
    purge_benign_generators_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm cleanup of only the five benign-generator roles",
    )
    purge_benign_generators_parser.set_defaults(
        handler=purge_benign_generators_command
    )

    # This command sends scripts/setup_victim.sh to the victim EC2 instance.
    # Then it runs that script on the victim.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli setup-victim
    setup_victim_parser = subparsers.add_parser("setup-victim")
    setup_victim_parser.set_defaults(handler=setup_victim)

    # This command sends scripts/setup_attacker.sh to the attacker EC2 instance.
    # Then it runs that script on the attacker.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli setup-attacker
    setup_attacker_parser = subparsers.add_parser("setup-attacker")
    setup_attacker_parser.set_defaults(handler=setup_attacker)

    # This command sends scripts/setup_ids.sh to the IDS EC2 instance.
    # Then it runs that script on the IDS machine.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli setup-ids
    setup_ids_parser = subparsers.add_parser("setup-ids")
    setup_ids_parser.set_defaults(handler=setup_ids)

    setup_benign_generators_parser = subparsers.add_parser(
        "setup-benign-generators",
        help="Install benign traffic tools on all five generators",
    )
    setup_benign_generators_parser.set_defaults(
        handler=setup_benign_generators
    )

    configure_benign_network_parser = subparsers.add_parser(
        "configure-benign-network",
        help="Synchronize victim and generator private IP settings",
    )
    configure_benign_network_parser.set_defaults(
        handler=configure_benign_network_command
    )

    verify_benign_connectivity_parser = subparsers.add_parser(
        "verify-benign-network",
        help="Test benign connectivity from all five generators",
    )
    verify_benign_connectivity_parser.set_defaults(
        handler=verify_benign_connectivity_command
    )

    deploy_benign_traffic_parser = subparsers.add_parser(
        "deploy-benign-traffic",
        help="Deploy the benign traffic agent to all generators",
    )
    deploy_benign_traffic_parser.set_defaults(
        handler=deploy_benign_traffic_command
    )

    verify_benign_traffic_parser = subparsers.add_parser(
        "verify-benign-traffic",
        help="Verify the benign traffic agent on all generators",
    )
    verify_benign_traffic_parser.set_defaults(
        handler=verify_benign_traffic_command
    )

    start_benign_traffic_parser = subparsers.add_parser(
        "start-benign-traffic",
        help="Start all benign generators for a fixed duration",
    )
    start_benign_traffic_parser.add_argument(
        "--duration-seconds",
        type=int,
        required=True,
    )
    start_benign_traffic_parser.set_defaults(
        handler=start_benign_traffic_command
    )

    stop_benign_traffic_parser = subparsers.add_parser(
        "stop-benign-traffic",
        help="Stop all benign generators",
    )
    stop_benign_traffic_parser.set_defaults(
        handler=stop_benign_traffic_command
    )

    benign_traffic_status_parser = subparsers.add_parser(
        "benign-traffic-status",
        help="Show benign generator status",
    )
    benign_traffic_status_parser.set_defaults(
        handler=benign_traffic_status_command
    )

    diagnose_benign_generators_parser = subparsers.add_parser(
        "diagnose-benign-generators",
        help="Run foreground diagnostics on all five benign generators",
    )
    diagnose_benign_generators_parser.add_argument(
        "--duration-seconds",
        type=int,
        default=10,
    )
    diagnose_benign_generators_parser.set_defaults(
        handler=diagnose_benign_generators_command
    )

    setup_victim_benign_services_parser = subparsers.add_parser(
        "setup-victim-benign-services",
        help="Configure victim-side benign IoT-like services",
    )
    setup_victim_benign_services_parser.set_defaults(
        handler=setup_victim_benign_services_command
    )

    verify_victim_benign_services_parser = subparsers.add_parser(
        "verify-victim-benign-services",
        help="Verify victim benign services and listening ports",
    )
    verify_victim_benign_services_parser.set_defaults(
        handler=verify_victim_benign_services_command
    )

    # This command reads the victim public IP from inventory.yaml.
    # Then it runs tcpdump on the victim for 20 seconds.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli capture-benign-http
    capture_benign_http_parser = subparsers.add_parser("capture-benign-http")
    capture_benign_http_parser.set_defaults(handler=capture_benign_http)

    # This command reads victim public IP from inventory.yaml.
    # Then it checks the benign HTTP PCAP file on the victim.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli verify-benign-pcap
    verify_benign_pcap_parser = subparsers.add_parser("verify-benign-pcap")
    verify_benign_pcap_parser.set_defaults(handler=verify_benign_pcap)

    # This command reads attacker public IP and victim private IP from inventory.yaml.
    # Then it runs ApacheBench from attacker to victim over the AWS private network.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli generate-benign-http
    generate_benign_http_parser = subparsers.add_parser("generate-benign-http")
    generate_benign_http_parser.set_defaults(handler=generate_benign_http_traffic)

    # This command generates selected lab traffic from attacker to victim.
    # Traffic code:
    #   1 = benign HTTP
    #   2 = controlled DoS HTTP flood
    #   3 = controlled DoS SYN flood
    # Terminal command examples:
    #   python3 -m aws_ids_testbed_10.cli generate-traffic 1 --requests 100 --concurrency 5
    #   python3 -m aws_ids_testbed_10.cli generate-traffic 2 --requests 1000 --concurrency 50
    #   python3 -m aws_ids_testbed_10.cli generate-traffic 3 --packet-count 1000 --port 80 --interval-microseconds 5000
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
    generate_traffic_parser.add_argument("--interval-microseconds", type=int)
    generate_traffic_parser.set_defaults(handler=generate_traffic_command)

    # This command saves victim target settings on the attacker.
    # It reads the victim private IP from inventory.yaml.
    # Then it writes /opt/aws_ids_testbed/config/attacker.env on the attacker.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli configure-attacker-victim-url
    configure_attacker_victim_parser = subparsers.add_parser(
        "configure-attacker-victim-url"
    )
    configure_attacker_victim_parser.set_defaults(handler=configure_attacker_victim)

    # This command verifies victim target settings saved on the attacker.
    # It reads the attacker public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli attacker-verify-victim-url
    attacker_verify_victim_url_parser = subparsers.add_parser(
        "attacker-verify-victim-url"
    )
    attacker_verify_victim_url_parser.set_defaults(
        handler=attacker_verify_victim_url_command
    )

    # This command copies the attacker traffic script to the attacker EC2 instance.
    # It reads the attacker public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli deploy-attacker-traffic
    deploy_attacker_traffic_parser = subparsers.add_parser("deploy-attacker-traffic")
    deploy_attacker_traffic_parser.set_defaults(handler=deploy_attacker_traffic_command)

    # This command verifies the attacker traffic script on the attacker EC2 instance.
    # It reads the attacker public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli verify-attacker-traffic
    verify_attacker_traffic_parser = subparsers.add_parser("verify-attacker-traffic")
    verify_attacker_traffic_parser.set_defaults(handler=verify_attacker_traffic_command)

    # This command runs attacker-side traffic generation.
    # The attacker reads victim target settings from attacker.env.
    # Traffic code:
    #   1 = benign HTTP
    #   2 = controlled DoS HTTP flood
    #   3 = controlled DoS SYN flood
    # Terminal command examples:
    #   python3 -m aws_ids_testbed_10.cli attacker-generate-traffic 1
    #   python3 -m aws_ids_testbed_10.cli attacker-generate-traffic 2 --requests 1000 --concurrency 50
    #   python3 -m aws_ids_testbed_10.cli attacker-generate-traffic 3 --packet-count 1000 --port 80 --interval-microseconds 5000
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
    attacker_generate_traffic_parser.add_argument("--interval-microseconds", type=int)
    attacker_generate_traffic_parser.set_defaults(
        handler=attacker_generate_traffic_command
    )

    # This command coordinates victim labeling/capture with attacker traffic generation.
    # It keeps the instances autonomous:
    #   victim reads ACTIVE_SCENARIO and captures/sends in the background
    #   attacker reads attacker.env and generates the selected traffic
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli run-scenario 1 --requests 100 --concurrency 5
    run_scenario_parser = subparsers.add_parser("run-scenario")
    run_scenario_parser.add_argument(
        "traffic_code",
        choices=["1", "2", "3", "4"],
        help=(
            "1=benign_http, 2=dos_http_flood, 3=dos_syn_flood, "
            "4=benign_iot_5_generators"
        ),
    )
    run_scenario_parser.add_argument("--requests", type=int)
    run_scenario_parser.add_argument("--concurrency", type=int)
    run_scenario_parser.add_argument("--packet-count", type=int)
    run_scenario_parser.add_argument("--port", type=int)
    run_scenario_parser.add_argument("--interval-microseconds", type=int)
    run_scenario_parser.add_argument(
        "--auto-stop-seconds",
        type=int,
        help=(
            "required positive duration; wait this many seconds after traffic "
            "generation, then stop capture, drain uploads, and stop the sender"
        ),
    )
    run_scenario_parser.set_defaults(handler=run_scenario_command)

    # This command copies only essential receiver files to the IDS EC2 instance.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli deploy-ids-receiver
    deploy_ids_receiver_parser = subparsers.add_parser("deploy-ids-receiver")
    deploy_ids_receiver_parser.set_defaults(handler=deploy_ids_receiver)

    # This command starts the IDS FastAPI receiver in the background.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-start-receiver
    ids_start_receiver_parser = subparsers.add_parser("ids-start-receiver")
    ids_start_receiver_parser.set_defaults(handler=ids_start_receiver_command)

    # This command lists PCAP files received by the IDS receiver.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-list-received-pcaps
    ids_list_received_pcaps_parser = subparsers.add_parser("ids-list-received-pcaps")
    ids_list_received_pcaps_parser.set_defaults(handler=ids_list_received_pcaps_command)

    # This command deploys the IDS-side PCAP-to-CSV converter and agent.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli deploy-ids-pcap-converter
    deploy_ids_pcap_converter_parser = subparsers.add_parser(
        "deploy-ids-pcap-converter"
    )
    deploy_ids_pcap_converter_parser.set_defaults(
        handler=deploy_ids_pcap_converter_command
    )

    # This command verifies the IDS-side PCAP-to-CSV converter and agent.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli verify-ids-pcap-converter
    verify_ids_pcap_converter_parser = subparsers.add_parser(
        "verify-ids-pcap-converter"
    )
    verify_ids_pcap_converter_parser.set_defaults(
        handler=verify_ids_pcap_converter_command
    )

    # This command starts the IDS-side PCAP-to-CSV converter agent.
    # It watches /home/ubuntu/aws_ids_testbed/input for new PCAP files.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-start-pcap-converter-agent
    ids_start_pcap_converter_agent_parser = subparsers.add_parser(
        "ids-start-pcap-converter-agent"
    )
    ids_start_pcap_converter_agent_parser.set_defaults(
        handler=start_ids_pcap_converter_agent_command
    )

    # This command stops the IDS-side PCAP-to-CSV converter agent.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-stop-pcap-converter-agent
    ids_stop_pcap_converter_agent_parser = subparsers.add_parser(
        "ids-stop-pcap-converter-agent"
    )
    ids_stop_pcap_converter_agent_parser.set_defaults(
        handler=stop_ids_pcap_converter_agent_command
    )

    # This command shows IDS-side PCAP-to-CSV converter agent status and logs.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-pcap-converter-agent-status
    ids_pcap_converter_agent_status_parser = subparsers.add_parser(
        "ids-pcap-converter-agent-status"
    )
    ids_pcap_converter_agent_status_parser.set_defaults(
        handler=ids_pcap_converter_agent_status_command
    )

    ids_delete_pcap_csv_parser = subparsers.add_parser(
        "ids-delete-pcap-csv"
    )
    ids_delete_pcap_csv_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Permanently delete IDS runtime PCAP and CSV files.",
    )
    ids_delete_pcap_csv_parser.set_defaults(
        handler=ids_delete_pcap_csv_command
    )

    # This command lists CSV files produced by IDS-side PCAP-to-CSV conversion.
    # It reads the IDS public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-list-csv-files
    ids_list_csv_files_parser = subparsers.add_parser("ids-list-csv-files")
    ids_list_csv_files_parser.set_defaults(handler=ids_list_csv_files_command)

    # This command downloads converted IDS CSV files to local Slurm.
    # By default, it downloads all CSV files from:
    #   /home/ubuntu/aws_ids_testbed/output/csv
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-download-csv-files
    ids_download_csv_files_parser = subparsers.add_parser("ids-download-csv-files")
    ids_download_csv_files_parser.add_argument(
        "--csv-path",
        help="Remote IDS CSV path or CSV filename. If omitted, downloads all CSV files.",
    )
    ids_download_csv_files_parser.add_argument(
        "--local-output-directory",
        help="Local output directory. Default: artifacts/ids_csv",
    )
    ids_download_csv_files_parser.set_defaults(handler=ids_download_csv_files_command)

    # This command runs local IDS Lite V6 inference STEP=1 on downloaded CSV files.
    # It creates the local capture manifest and loaded-data pickle for later
    # preprocessing/windowing/inference steps.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-run-inference-step1
    ids_run_inference_step1_parser = subparsers.add_parser("ids-run-inference-step1")
    ids_run_inference_step1_parser.add_argument(
        "--input-directory",
        help="Local IDS CSV input directory. Default: artifacts/ids_csv",
    )
    ids_run_inference_step1_parser.add_argument(
        "--output-path",
        help=(
            "Output pickle path. Default: "
            "artifacts/ids_inference/step_1_Loaded_IDS_Data.pkl"
        ),
    )
    ids_run_inference_step1_parser.add_argument(
        "--manifest-path",
        help=(
            "Output manifest CSV path. Default: "
            "artifacts/ids_inference/ids_capture_manifest.csv"
        ),
    )
    ids_run_inference_step1_parser.set_defaults(
        handler=ids_run_inference_step1_command
    )

    # This command runs local IDS Lite V6 inference STEP=2 preprocessing.
    # It reads the local STEP=1 pickle and saves cleaned STEP=2 data.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-run-inference-step2
    ids_run_inference_step2_parser = subparsers.add_parser("ids-run-inference-step2")
    ids_run_inference_step2_parser.add_argument(
        "--input-path",
        help=(
            "Input pickle path. Default: "
            "artifacts/ids_inference/step_1_Loaded_IDS_Data.pkl"
        ),
    )
    ids_run_inference_step2_parser.add_argument(
        "--output-path",
        help=(
            "Output pickle path. Default: "
            "artifacts/ids_inference/step_2_Processed_IDS_Data.pkl"
        ),
    )
    ids_run_inference_step2_parser.set_defaults(
        handler=ids_run_inference_step2_command
    )

    ids_run_inference_step3_parser = subparsers.add_parser("ids-run-inference-step3")
    ids_run_inference_step3_parser.add_argument("--input-path")
    ids_run_inference_step3_parser.add_argument("--window-output-path")
    ids_run_inference_step3_parser.add_argument("--metadata-output-path")
    ids_run_inference_step3_parser.add_argument("--features-output-path")
    ids_run_inference_step3_parser.add_argument("--summary-output-path")
    ids_run_inference_step3_parser.set_defaults(
        handler=ids_run_inference_step3_command
    )

    # This command runs local IDS Lite V6 inference STEP=4 prediction.
    # It reads STEP=3 windows, loads the saved Lite V6 model, and saves
    # one prediction row per IDS window.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-run-inference-step4
    ids_run_inference_step4_parser = subparsers.add_parser("ids-run-inference-step4")
    ids_run_inference_step4_parser.add_argument(
        "--window-data-path",
        help=(
            "Input NumPy window path. Default: "
            "artifacts/ids_inference/X_IDS_windows.npy"
        ),
    )
    ids_run_inference_step4_parser.add_argument(
        "--metadata-path",
        help=(
            "Input window metadata CSV path. Default: "
            "artifacts/ids_inference/ids_window_metadata.csv"
        ),
    )
    ids_run_inference_step4_parser.add_argument(
        "--predictions-output-path",
        help=(
            "Output prediction CSV path. Default: "
            "artifacts/ids_inference/ids_predictions.csv"
        ),
    )
    ids_run_inference_step4_parser.add_argument(
        "--summary-output-path",
        help=(
            "Output prediction summary pickle path. Default: "
            "artifacts/ids_inference/ids_prediction_summary.pkl"
        ),
    )
    ids_run_inference_step4_parser.add_argument(
        "--batch-size",
        type=int,
        help="Prediction batch size. Default: IDS_PREDICTION_BATCH_SIZE from config.",
    )
    ids_run_inference_step4_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Prediction device. Default: IDS_PREDICTION_DEVICE from config.",
    )
    ids_run_inference_step4_parser.set_defaults(
        handler=ids_run_inference_step4_command
    )

    # This command prints a read-only debug report for IDS Lite V6 artifacts.
    # It checks paths, feature counts, scaler shape, label mapping, and window
    # metadata alignment without changing any files.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-debug-inference-artifacts
    ids_debug_inference_artifacts_parser = subparsers.add_parser(
        "ids-debug-inference-artifacts"
    )
    ids_debug_inference_artifacts_parser.add_argument("--step-2-data-path")
    ids_debug_inference_artifacts_parser.add_argument("--window-data-path")
    ids_debug_inference_artifacts_parser.add_argument("--metadata-path")
    ids_debug_inference_artifacts_parser.add_argument("--features-used-path")
    ids_debug_inference_artifacts_parser.set_defaults(
        handler=ids_debug_inference_artifacts_command
    )

    # This command prints a read-only distribution report for selected IDS features.
    # It compares raw AWS IDS feature values with the saved Lite V6 scaler.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-debug-feature-distribution
    ids_debug_feature_distribution_parser = subparsers.add_parser(
        "ids-debug-feature-distribution"
    )
    ids_debug_feature_distribution_parser.add_argument("--step-2-data-path")
    ids_debug_feature_distribution_parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top feature rows to show in each debug section.",
    )
    ids_debug_feature_distribution_parser.set_defaults(
        handler=ids_debug_feature_distribution_command
    )

    ids_debug_selected_feature_values_parser = subparsers.add_parser(
        "ids-debug-selected-feature-values"
    )
    ids_debug_selected_feature_values_parser.add_argument("--step-2-data-path")
    ids_debug_selected_feature_values_parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature names to inspect from STEP=2 data.",
    )
    ids_debug_selected_feature_values_parser.set_defaults(
        handler=ids_debug_selected_feature_values_command
    )

    # This command prints top-k model probabilities from saved IDS predictions.
    # It helps show whether the model strongly prefers the top class or only
    # barely ranks it above other labels.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-debug-prediction-probabilities
    ids_debug_prediction_probabilities_parser = subparsers.add_parser(
        "ids-debug-prediction-probabilities"
    )
    ids_debug_prediction_probabilities_parser.add_argument("--predictions-path")
    ids_debug_prediction_probabilities_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    ids_debug_prediction_probabilities_parser.add_argument(
        "--sample-windows-per-file",
        type=int,
        default=5,
    )
    ids_debug_prediction_probabilities_parser.set_defaults(
        handler=ids_debug_prediction_probabilities_command
    )

    # This command compares AWS IDS model-ready windows with Lite V6
    # model-ready class windows from train/val/test saved artifacts.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-debug-compare-lite-v6-classes
    ids_debug_compare_lite_v6_classes_parser = subparsers.add_parser(
        "ids-debug-compare-lite-v6-classes"
    )
    ids_debug_compare_lite_v6_classes_parser.add_argument("--window-data-path")
    ids_debug_compare_lite_v6_classes_parser.add_argument("--metadata-path")
    ids_debug_compare_lite_v6_classes_parser.add_argument(
        "--labels",
        nargs="+",
        help="Lite V6 labels to compare. Default: config IDS_DEBUG_COMPARE_LABELS.",
    )
    ids_debug_compare_lite_v6_classes_parser.add_argument(
        "--sample-windows-per-label",
        type=int,
        help="Maximum Lite V6 windows sampled per label.",
    )
    ids_debug_compare_lite_v6_classes_parser.add_argument(
        "--top-n-features",
        type=int,
        default=15,
        help="Number of most different features to show per matching label.",
    )
    ids_debug_compare_lite_v6_classes_parser.add_argument(
        "--source-file-prefix",
        dest="source_file_prefixes",
        action="append",
        help=(
            "Compare only AWS windows whose source filename starts with this "
            "prefix. Repeat the option to select multiple prefixes."
        ),
    )
    ids_debug_compare_lite_v6_classes_parser.set_defaults(
        handler=ids_debug_compare_lite_v6_classes_command
    )

    # This command manually converts one PCAP file already saved on the IDS.
    # It is mainly for testing before automatic conversion is enabled.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-convert-pcap --pcap-path PATH
    ids_convert_pcap_parser = subparsers.add_parser("ids-convert-pcap")
    ids_convert_pcap_parser.add_argument("--pcap-path", required=True)
    ids_convert_pcap_parser.set_defaults(handler=ids_convert_pcap_command)

    # This command manually converts all PCAP files already saved on the IDS.
    # It processes one PCAP file fully before moving to the next one.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli ids-convert-received-pcaps
    ids_convert_received_pcaps_parser = subparsers.add_parser(
        "ids-convert-received-pcaps"
    )
    ids_convert_received_pcaps_parser.set_defaults(
        handler=ids_convert_received_pcaps_command
    )

    # This command saves IDS receiver settings on the victim.
    # It reads the IDS private IP from inventory.yaml.
    # Then it writes /opt/aws_ids_testbed/config/victim.env on the victim.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli configure-victim-ids-url
    configure_victim_ids_parser = subparsers.add_parser("configure-victim-ids-url")
    configure_victim_ids_parser.set_defaults(handler=configure_victim_ids)

    # This command verifies IDS receiver settings saved on the victim.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-verify-ids-url
    victim_verify_ids_url_parser = subparsers.add_parser("victim-verify-ids-url")
    victim_verify_ids_url_parser.set_defaults(handler=victim_verify_ids_url_command)

    # This command updates ACTIVE_SCENARIO in victim.env.
    # The future continuous capture agent will use this label when naming PCAP chunks.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-set-scenario benign_http
    victim_set_scenario_parser = subparsers.add_parser("victim-set-scenario")
    victim_set_scenario_parser.add_argument(
        "scenario",
        choices=["benign_http", "dos_http_flood", "dos_syn_flood"],
    )
    victim_set_scenario_parser.set_defaults(handler=victim_set_scenario_command)

    # This command copies the victim PCAP sender script to the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli deploy-victim-sender
    deploy_victim_sender_parser = subparsers.add_parser("deploy-victim-sender")
    deploy_victim_sender_parser.set_defaults(handler=deploy_victim_sender_command)

    # This command verifies the victim PCAP sender script on the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli verify-victim-sender
    verify_victim_sender_parser = subparsers.add_parser("verify-victim-sender")
    verify_victim_sender_parser.set_defaults(handler=verify_victim_sender_command)

    # This command sends any completed PCAP file from victim to IDS.
    # It reads the victim public IP from inventory.yaml.
    # The victim sender script reads IDS_RECEIVER_URL from victim.env.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-send-pcap --pcap-path PATH --scenario LABEL
    victim_send_pcap_parser = subparsers.add_parser("victim-send-pcap")
    victim_send_pcap_parser.add_argument("--pcap-path", required=True)
    victim_send_pcap_parser.add_argument("--scenario", default="unknown")
    victim_send_pcap_parser.set_defaults(handler=victim_send_pcap_command)

    # This command sends all completed pending PCAP files from victim to IDS.
    # It does not require a hard-coded PCAP filename or scenario.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-send-pending-pcaps
    victim_send_pending_pcaps_parser = subparsers.add_parser(
        "victim-send-pending-pcaps"
    )
    victim_send_pending_pcaps_parser.set_defaults(
        handler=victim_send_pending_pcaps_command
    )

    # This command copies the pending PCAP sender agent to the victim EC2 instance.
    # The agent will later upload pending PCAP files to IDS in the background.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli deploy-victim-sender-agent
    deploy_victim_sender_agent_parser = subparsers.add_parser(
        "deploy-victim-sender-agent"
    )
    deploy_victim_sender_agent_parser.set_defaults(
        handler=deploy_victim_sender_agent_command
    )

    # This command verifies the pending PCAP sender agent on the victim EC2 instance.
    # It checks that the script exists, is executable, and has valid bash syntax.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli verify-victim-sender-agent
    verify_victim_sender_agent_parser = subparsers.add_parser(
        "verify-victim-sender-agent"
    )
    verify_victim_sender_agent_parser.set_defaults(
        handler=verify_victim_sender_agent_command
    )

    # This command starts the pending PCAP sender agent in the background.
    # It writes logs to /opt/aws_ids_testbed/logs/pending_pcap_sender_agent.log.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-start-sender-agent
    victim_start_sender_agent_parser = subparsers.add_parser(
        "victim-start-sender-agent"
    )
    victim_start_sender_agent_parser.set_defaults(
        handler=start_victim_sender_agent_command
    )

    # This command stops the pending PCAP sender agent.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-stop-sender-agent
    victim_stop_sender_agent_parser = subparsers.add_parser(
        "victim-stop-sender-agent"
    )
    victim_stop_sender_agent_parser.set_defaults(
        handler=stop_victim_sender_agent_command
    )

    # This command shows whether the pending PCAP sender agent is running.
    # It also prints recent sender-agent log lines.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-sender-agent-status
    victim_sender_agent_status_parser = subparsers.add_parser(
        "victim-sender-agent-status"
    )
    victim_sender_agent_status_parser.set_defaults(
        handler=victim_sender_agent_status_command
    )

    verify_victim_sender_delivery_parser = subparsers.add_parser(
        "verify-victim-sender-delivery",
        help="Verify that victim PCAP files were delivered",
    )
    verify_victim_sender_delivery_parser.set_defaults(
        handler=verify_victim_sender_delivery_command
    )

    # This command starts both victim background agents.
    # It starts:
    #   1. continuous capture agent
    #   2. pending PCAP sender agent
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-start-agents
    victim_start_agents_parser = subparsers.add_parser("victim-start-agents")
    victim_start_agents_parser.set_defaults(handler=victim_start_agents_command)

    # This command stops both victim background agents.
    # It stops:
    #   1. pending PCAP sender agent
    #   2. continuous capture agent
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-stop-agents
    victim_stop_agents_parser = subparsers.add_parser("victim-stop-agents")
    victim_stop_agents_parser.set_defaults(handler=victim_stop_agents_command)

    # This command shows both victim background agent statuses.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-agents-status
    victim_agents_status_parser = subparsers.add_parser("victim-agents-status")
    victim_agents_status_parser.set_defaults(handler=victim_agents_status_command)

    # This command copies the victim scenario capture script to the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli deploy-victim-capture
    deploy_victim_capture_parser = subparsers.add_parser("deploy-victim-capture")
    deploy_victim_capture_parser.set_defaults(handler=deploy_victim_capture_command)

    # This command verifies the victim scenario capture script on the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli verify-victim-capture
    verify_victim_capture_parser = subparsers.add_parser("verify-victim-capture")
    verify_victim_capture_parser.set_defaults(handler=verify_victim_capture_command)

    # This command captures one scenario on the victim EC2 instance.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-capture-scenario --scenario LABEL --seconds 20
    victim_capture_scenario_parser = subparsers.add_parser("victim-capture-scenario")
    victim_capture_scenario_parser.add_argument("--scenario", required=True)
    victim_capture_scenario_parser.add_argument("--seconds", type=int, default=20)
    victim_capture_scenario_parser.set_defaults(handler=victim_capture_scenario_command)

    # This command lists victim PCAP files in writing, pending, sent, and failed.
    # It reads the victim public IP from inventory.yaml.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-list-pcaps
    victim_list_pcaps_parser = subparsers.add_parser("victim-list-pcaps")
    victim_list_pcaps_parser.set_defaults(handler=victim_list_pcaps_command)

    victim_delete_pcap_csv_parser = subparsers.add_parser(
        "victim-delete-pcap-csv"
    )
    victim_delete_pcap_csv_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Permanently delete victim runtime PCAP and CSV files.",
    )
    victim_delete_pcap_csv_parser.set_defaults(
        handler=victim_delete_pcap_csv_command
    )

    # This command verifies the expected benign protocols in the newest
    # pending PCAP on the victim.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli verify-benign-pcap-protocols
    verify_benign_pcap_protocols_parser = subparsers.add_parser(
        "verify-benign-pcap-protocols"
    )
    verify_benign_pcap_protocols_parser.set_defaults(
        handler=verify_benign_pcap_protocols_command
    )

    # This command copies the continuous capture agent to the victim EC2 instance.
    # The agent will later capture 10-second PCAP chunks in the background.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli deploy-victim-capture-agent
    deploy_victim_capture_agent_parser = subparsers.add_parser(
        "deploy-victim-capture-agent"
    )
    deploy_victim_capture_agent_parser.set_defaults(
        handler=deploy_victim_capture_agent_command
    )

    # This command verifies the continuous capture agent on the victim EC2 instance.
    # It checks that the script exists, is executable, and has valid bash syntax.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli verify-victim-capture-agent
    verify_victim_capture_agent_parser = subparsers.add_parser(
        "verify-victim-capture-agent"
    )
    verify_victim_capture_agent_parser.set_defaults(
        handler=verify_victim_capture_agent_command
    )

    # This command starts the continuous capture agent in the background.
    # It writes logs to /opt/aws_ids_testbed/logs/continuous_capture_agent.log.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-start-capture-agent
    victim_start_capture_agent_parser = subparsers.add_parser(
        "victim-start-capture-agent"
    )
    victim_start_capture_agent_parser.set_defaults(
        handler=start_victim_capture_agent_command
    )

    # This command stops the continuous capture agent.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-stop-capture-agent
    victim_stop_capture_agent_parser = subparsers.add_parser(
        "victim-stop-capture-agent"
    )
    victim_stop_capture_agent_parser.set_defaults(
        handler=stop_victim_capture_agent_command
    )

    # This command shows whether the continuous capture agent is running.
    # It also prints recent agent log lines.
    # Terminal command:
    #   python3 -m aws_ids_testbed_10.cli victim-capture-agent-status
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
