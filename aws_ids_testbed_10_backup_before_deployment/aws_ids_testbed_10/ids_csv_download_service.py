"""Download converted IDS CSV files from the IDS EC2 instance."""

from __future__ import annotations

import posixpath
import stat
from pathlib import Path

from aws_ids_testbed_10.remote_settings import (
    get_private_key_path,
    get_public_host,
    get_ssh_username,
)


REMOTE_CSV_DIR = "/home/ubuntu/aws_ids_testbed/output/csv"
DEFAULT_LOCAL_CSV_DIR = "artifacts/ids_csv"


def download_ids_csv_files(
    project_root: Path,
    csv_path: str | None = None,
    local_output_directory: str | None = None,
) -> int:
    """Download one or all converted CSV files from IDS to local Slurm."""
    import paramiko

    username = get_ssh_username(project_root)
    private_key_path = get_private_key_path(project_root)
    ids_public_host = get_public_host(project_root, "ids")
    local_output_dir = _resolve_local_output_dir(
        project_root=project_root,
        local_output_directory=local_output_directory,
    )
    local_output_dir.mkdir(parents=True, exist_ok=True)

    if not private_key_path.exists():
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

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

    try:
        with ssh_client.open_sftp() as sftp:
            remote_csv_paths = _get_remote_csv_paths(
                sftp=sftp,
                csv_path=csv_path,
            )

            if not remote_csv_paths:
                print("[ids-csv-download] No CSV files found on IDS.")
                return 0

            print(f"[ids-csv-download] Local output: {local_output_dir}")
            for remote_csv_path in remote_csv_paths:
                filename = posixpath.basename(remote_csv_path)
                local_csv_path = local_output_dir / filename
                print(f"[ids-csv-download] Downloading: {remote_csv_path}")
                sftp.get(remote_csv_path, str(local_csv_path))
                print(f"[ids-csv-download] Saved: {local_csv_path}")

            print(f"[ids-csv-download] Downloaded CSV files: {len(remote_csv_paths)}")
            return 0

    finally:
        ssh_client.close()


def _resolve_local_output_dir(
    project_root: Path,
    local_output_directory: str | None,
) -> Path:
    """Return absolute local directory for downloaded IDS CSV files."""
    if local_output_directory:
        output_dir = Path(local_output_directory).expanduser()
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        return output_dir

    return project_root / DEFAULT_LOCAL_CSV_DIR


def _get_remote_csv_paths(sftp: object, csv_path: str | None) -> list[str]:
    """Return one requested CSV path or all CSV paths from the IDS CSV folder."""
    if csv_path:
        remote_csv_path = _normalize_remote_csv_path(csv_path)
        sftp.stat(remote_csv_path)
        return [remote_csv_path]

    try:
        entries = sftp.listdir_attr(REMOTE_CSV_DIR)
    except FileNotFoundError:
        return []

    csv_paths = []
    for entry in entries:
        if stat.S_ISREG(entry.st_mode) and entry.filename.endswith(".csv"):
            csv_paths.append(f"{REMOTE_CSV_DIR}/{entry.filename}")

    return sorted(csv_paths)


def _normalize_remote_csv_path(csv_path: str) -> str:
    """Accept a full IDS CSV path or only a CSV filename."""
    cleaned_path = csv_path.strip()

    if not cleaned_path.endswith(".csv"):
        raise ValueError("CSV path must end with .csv")

    if "/" not in cleaned_path:
        return f"{REMOTE_CSV_DIR}/{cleaned_path}"

    expected_prefix = f"{REMOTE_CSV_DIR}/"
    if not cleaned_path.startswith(expected_prefix):
        raise ValueError(
            "CSV path must be inside "
            f"{REMOTE_CSV_DIR} or be only a CSV filename."
        )

    return cleaned_path
