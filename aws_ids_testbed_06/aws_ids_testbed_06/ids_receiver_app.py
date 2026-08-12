"""Small FastAPI receiver that accepts PCAP uploads on the IDS instance."""

from __future__ import annotations

from pathlib import Path

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

    destination = IDS_INPUT_DIR / file.filename

    with destination.open("wb") as output_file:
        while chunk := await file.read(1024 * 1024):
            output_file.write(chunk)

    return {
        "status": "saved",
        "filename": file.filename,
        "path": str(destination),
    }
