"""Small FastAPI receiver that accepts PCAP uploads on the IDS instance."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile


IDS_INPUT_DIR = Path("/home/ubuntu/aws_ids_testbed/input")

app = FastAPI(title="AWS IDS Testbed Receiver")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple health check response."""
    return {"status": "ok"}


@app.post("/upload-pcap")
async def upload_pcap(file: UploadFile = File(...)) -> dict[str, str]:
    """Receive one PCAP file and save it on the IDS instance."""
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

    return {
        "status": "saved",
        "filename": safe_filename,
        "path": str(destination),
    }
