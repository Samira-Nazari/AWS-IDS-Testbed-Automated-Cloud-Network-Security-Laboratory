"""Download generated IDS timing reports from the IDS EC2 instance."""

from __future__ import annotations

import posixpath
from pathlib import Path

from aws_ids_testbed_11.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


REMOTE_TIMING_REPORT_DIR = (
    "/home/ubuntu/aws_ids_testbed/inference"
)
TIMING_REPORT_NAMES = (
    "ids_pcap_timing.csv",
    "ids_transaction_timing.csv",
    "ids_window_timing.csv",
)
DEFAULT_LOCAL_TIMING_REPORT_DIR = "artifacts/ids_timing"
REMOTE_PREDICTIONS_PATH = (
    "/home/ubuntu/aws_ids_testbed/inference/ids_predictions.csv"
)
DEFAULT_LOCAL_PREDICTION_DIR = "artifacts/ids_inference"


def download_ids_timing_reports(
    project_root: Path,
    local_output_directory: str | None = None,
) -> int:
    """Download available timing reports to local Slurm storage."""

    import paramiko

    project_root = Path(project_root)
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    if not private_key_path.exists():
        raise FileNotFoundError(
            f"Private key not found: {private_key_path}"
        )

    local_output_dir = Path(
        local_output_directory or DEFAULT_LOCAL_TIMING_REPORT_DIR
    ).expanduser()
    if not local_output_dir.is_absolute():
        local_output_dir = project_root / local_output_dir
    local_output_dir.mkdir(parents=True, exist_ok=True)

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {username}@{ids_public_host} ...")
    ssh_client.connect(
        hostname=ids_public_host,
        username=username,
        key_filename=str(private_key_path),
        timeout=30,
    )

    downloaded = 0
    try:
        with ssh_client.open_sftp() as sftp:
            for report_name in TIMING_REPORT_NAMES:
                remote_path = posixpath.join(
                    REMOTE_TIMING_REPORT_DIR,
                    report_name,
                )
                local_path = local_output_dir / report_name
                temporary_path = local_path.with_name(
                    f".{report_name}.tmp"
                )

                try:
                    sftp.stat(remote_path)
                except FileNotFoundError:
                    print(
                        f"[ids-timing-download] Missing on IDS: "
                        f"{remote_path}"
                    )
                    continue

                try:
                    print(
                        f"[ids-timing-download] Downloading: "
                        f"{remote_path}"
                    )
                    sftp.get(remote_path, str(temporary_path))
                    temporary_path.replace(local_path)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()

                print(f"[ids-timing-download] Saved: {local_path}")
                downloaded += 1
    finally:
        ssh_client.close()

    if downloaded == 0:
        print("[ids-timing-download] No timing reports found on IDS.")
        return 0

    print(
        f"[ids-timing-download] Downloaded reports: {downloaded}; "
        f"directory: {local_output_dir}"
    )
    return 0


def download_ids_prediction_csv(
    project_root: Path,
    local_output_directory: str | None = None,
) -> int:
    """Download the cumulative IDS prediction CSV to local Slurm."""

    import paramiko

    project_root = Path(project_root)
    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")

    if not private_key_path.exists():
        raise FileNotFoundError(
            f"Private key not found: {private_key_path}"
        )

    local_output_dir = Path(
        local_output_directory or DEFAULT_LOCAL_PREDICTION_DIR
    ).expanduser()
    if not local_output_dir.is_absolute():
        local_output_dir = project_root / local_output_dir
    local_output_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_output_dir / "ids_predictions.csv"
    temporary_path = local_output_dir / ".ids_predictions.csv.tmp"

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"Connecting to {username}@{ids_public_host} ...")
        ssh_client.connect(
            hostname=ids_public_host,
            username=username,
            key_filename=str(private_key_path),
            timeout=30,
        )

        with ssh_client.open_sftp() as sftp:
            sftp.stat(REMOTE_PREDICTIONS_PATH)
            print(
                "[ids-prediction-download] Downloading: "
                f"{REMOTE_PREDICTIONS_PATH}"
            )
            sftp.get(
                REMOTE_PREDICTIONS_PATH,
                str(temporary_path),
            )
            temporary_path.replace(local_path)

        print(f"[ids-prediction-download] Saved: {local_path}")
        return 0
    except Exception as exception:
        print(
            "[ids-prediction-download] Download failed: "
            f"{type(exception).__name__}: {exception}"
        )
        return 1
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        ssh_client.close()
