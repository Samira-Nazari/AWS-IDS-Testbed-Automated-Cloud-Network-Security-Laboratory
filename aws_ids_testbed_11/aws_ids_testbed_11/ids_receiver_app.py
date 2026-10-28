"""Small FastAPI receiver that accepts PCAP uploads on the IDS instance."""

from __future__ import annotations

import os
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile

try:
    # The deployed receiver runs with /home/ubuntu/aws_ids_testbed on sys.path.
    from ids_timing import append_timing_event, timing_events_path, utc_now_iso
except ModuleNotFoundError:  # pragma: no cover - used by local package tests
    from aws_ids_testbed_11.ids_timing import (
        append_timing_event,
        timing_events_path,
        utc_now_iso,
    )


IDS_BASE_DIR = Path("/home/ubuntu/aws_ids_testbed")
IDS_INPUT_DIR = Path("/home/ubuntu/aws_ids_testbed/input")
IDS_TIMING_EVENTS_PATH = timing_events_path(IDS_BASE_DIR)

app = FastAPI(title="AWS IDS Testbed Receiver")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple health check response."""
    return {"status": "ok"}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one received PCAP."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.post("/upload-pcap")
async def upload_pcap(
    file: UploadFile = File(...),
    source_role: str = Form("unknown"),
    scenario: str = Form("unknown"),
    timing_metadata: str = Form("{}"),
    upload_started_at_utc: str = Form(""),
) -> dict[str, str]:
    """Receive one PCAP file and save it on the IDS instance."""
    receive_started_at_utc = utc_now_iso()
    receive_started_monotonic_ns = time.monotonic_ns()

    IDS_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename or "upload.pcap").name
    destination = IDS_INPUT_DIR / safe_filename
    temporary_destination = IDS_INPUT_DIR / (
        f".{safe_filename}.{uuid4().hex}.uploading"
    )

    try:
        with temporary_destination.open("wb") as output_file:
            while chunk := await file.read(1024 * 1024):
                output_file.write(chunk)

        os.replace(temporary_destination, destination)
    finally:
        if temporary_destination.exists():
            temporary_destination.unlink()

    received_at_utc = utc_now_iso()
    receive_duration_ms = round(
        (time.monotonic_ns() - receive_started_monotonic_ns) / 1_000_000,
        3,
    )

    try:
        parsed_metadata = json.loads(timing_metadata or "{}")
        metadata = (
            dict(parsed_metadata)
            if isinstance(parsed_metadata, dict)
            else {"timing_metadata_parse_error": True}
        )
    except json.JSONDecodeError:
        metadata = {"timing_metadata_parse_error": True}

    event = dict(metadata)
    event.update(
        {
            "event_type": "pcap_received",
            "capture_id": str(
                event.get("capture_id") or Path(safe_filename).stem
            ),
            "pcap_filename": safe_filename,
            "scenario": scenario,
            "source_role": source_role,
            "upload_started_at_utc": upload_started_at_utc,
            "receive_started_at_utc": receive_started_at_utc,
            "received_at_utc": received_at_utc,
            "receive_duration_ms": receive_duration_ms,
            "pcap_size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
    )

    metadata_path = destination.with_name(f"{destination.name}.timing.json")
    temporary_metadata_path = metadata_path.with_name(
        f".{metadata_path.name}.{uuid4().hex}.tmp"
    )
    try:
        temporary_metadata_path.write_text(
            json.dumps(event, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata_path, metadata_path)
    finally:
        if temporary_metadata_path.exists():
            temporary_metadata_path.unlink()

    append_timing_event(IDS_TIMING_EVENTS_PATH, event)

    return {
        "status": "saved",
        "filename": safe_filename,
        "path": str(destination),
        "received_at_utc": received_at_utc,
    }
