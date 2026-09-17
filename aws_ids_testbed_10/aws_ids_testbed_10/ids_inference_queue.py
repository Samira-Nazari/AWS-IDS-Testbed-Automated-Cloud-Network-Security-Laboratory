"""Build paths for queued IDS inference jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from aws_ids_testbed_10.ids_inference_workspace import (
    validate_runtime_name,
)


@dataclass(frozen=True)
class InferenceQueueJobPaths:
    """Files belonging to one queued inference job."""

    csv_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class InferenceQueueReservation:
    """Reserved paths for one inference queue job."""

    sequence: int
    scenario: str
    transaction_id: str
    job_paths: InferenceQueueJobPaths
    staging_csv_path: Path


def allocate_next_queue_sequence(
    *,
    queue_directory: Path,
) -> int:
    """Atomically allocate the next inference queue sequence."""

    queue_directory = Path(queue_directory)
    queue_directory.mkdir(parents=True, exist_ok=True)

    sequence_path = queue_directory / ".last_sequence"
    temporary_path = queue_directory / ".last_sequence.tmp"

    last_sequence = 0

    if sequence_path.is_file():
        sequence_text = sequence_path.read_text(
            encoding="utf-8"
        ).strip()

        if sequence_text:
            last_sequence = int(sequence_text)

        if last_sequence < 0:
            raise ValueError(
                "Stored queue sequence cannot be negative"
            )

    next_sequence = last_sequence + 1

    temporary_path.write_text(
        f"{next_sequence}\n",
        encoding="utf-8",
    )
    temporary_path.replace(sequence_path)

    return next_sequence


def build_inference_queue_job_paths(
    *,
    queue_directory: Path,
    scenario: str,
    sequence: int,
    transaction_id: str,
) -> InferenceQueueJobPaths:
    """Build scenario-first queue filenames."""

    scenario = validate_runtime_name(
        scenario,
        field_name="scenario",
    )
    transaction_id = validate_runtime_name(
        transaction_id,
        field_name="transaction_id",
    )

    if sequence <= 0:
        raise ValueError("sequence must be positive")

    scenario_prefix = f"{scenario}_"

    if transaction_id.startswith(scenario_prefix):
        source_suffix = transaction_id[
            len(scenario_prefix):
        ]
    else:
        source_suffix = transaction_id

    job_stem = (
        f"{scenario}_"
        f"{sequence:06d}_"
        f"{source_suffix}_"
        "inference"
    )

    queue_directory = Path(queue_directory)

    return InferenceQueueJobPaths(
        csv_path=queue_directory / f"{job_stem}.csv",
        manifest_path=queue_directory / f"{job_stem}.json",
    )


def reserve_inference_queue_job(
    *,
    queue_directory: Path,
    scenario: str,
    transaction_id: str,
) -> InferenceQueueReservation:
    """Reserve the next scenario-first queue filename."""

    sequence = allocate_next_queue_sequence(
        queue_directory=queue_directory,
    )

    job_paths = build_inference_queue_job_paths(
        queue_directory=queue_directory,
        scenario=scenario,
        sequence=sequence,
        transaction_id=transaction_id,
    )

    staging_csv_path = job_paths.csv_path.with_name(
        f".{job_paths.csv_path.name}.preparing"
    )

    if (
        staging_csv_path.exists()
        or job_paths.csv_path.exists()
        or job_paths.manifest_path.exists()
    ):
        raise FileExistsError(
            "Reserved inference queue path already exists"
        )

    return InferenceQueueReservation(
        sequence=sequence,
        scenario=scenario,
        transaction_id=transaction_id,
        job_paths=job_paths,
        staging_csv_path=staging_csv_path,
    )


def publish_inference_queue_job(
    *,
    reservation: InferenceQueueReservation,
    metadata: Mapping[str, object],
) -> InferenceQueueJobPaths:
    """Publish a completely prepared inference queue job."""

    staging_csv_path = reservation.staging_csv_path
    final_csv_path = reservation.job_paths.csv_path
    manifest_path = reservation.job_paths.manifest_path

    if not staging_csv_path.is_file():
        raise FileNotFoundError(
            f"Staging CSV does not exist: {staging_csv_path}"
        )

    if staging_csv_path.stat().st_size == 0:
        raise ValueError(
            f"Staging CSV is empty: {staging_csv_path}"
        )

    if final_csv_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Final inference queue job already exists"
        )

    temporary_manifest_path = manifest_path.with_name(
        f".{manifest_path.name}.preparing"
    )

    payload = dict(metadata)
    payload.update(
        {
            "status": "ready",
            "sequence": reservation.sequence,
            "scenario": reservation.scenario,
            "transaction_id": reservation.transaction_id,
            "queue_csv_path": str(final_csv_path),
        }
    )

    try:
        # Publish the complete CSV first.
        staging_csv_path.replace(final_csv_path)

        temporary_manifest_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # Publishing the manifest makes the job visible to Agent 2.
        temporary_manifest_path.replace(manifest_path)

    except Exception:
        temporary_manifest_path.unlink(missing_ok=True)

        # Restore the CSV to its staging location so publication
        # can be retried safely.
        if (
            final_csv_path.exists()
            and not staging_csv_path.exists()
            and not manifest_path.exists()
        ):
            final_csv_path.replace(staging_csv_path)

        raise

    return reservation.job_paths
