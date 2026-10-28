"""Shared timing utilities for the AWS IDS pipeline."""

from __future__ import annotations

import fcntl
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def timing_events_path(base_directory: Path) -> Path:
    """Return the central append-only timing event log."""

    return Path(base_directory) / "logs" / "ids_timing_events.jsonl"


def utc_now_iso() -> str:
    """Return UTC time with microsecond precision."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class TimingSpan:
    """Measure one processing stage."""

    stage: str
    started_at_utc: str
    started_monotonic_ns: int

    @classmethod
    def start(cls, stage: str) -> "TimingSpan":
        if not stage:
            raise ValueError("stage must not be empty")

        return cls(
            stage=stage,
            started_at_utc=utc_now_iso(),
            started_monotonic_ns=time.monotonic_ns(),
        )

    def finish(self) -> dict[str, object]:
        """Return completion timestamps and elapsed milliseconds."""

        return {
            "timing_stage": self.stage,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": utc_now_iso(),
            "duration_ms": round(
                (
                    time.monotonic_ns()
                    - self.started_monotonic_ns
                )
                / 1_000_000,
                3,
            ),
        }


def append_timing_event(
    events_path: Path,
    event: Mapping[str, object],
) -> None:
    """Append one structured timing event safely."""

    payload = dict(event)
    payload.setdefault("event_at_utc", utc_now_iso())
    payload.setdefault("timing_schema_version", 1)

    if "event_type" not in payload:
        raise ValueError("timing event must contain event_type")

    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    with events_path.open("a", encoding="utf-8") as event_file:
        fcntl.flock(event_file.fileno(), fcntl.LOCK_EX)

        try:
            event_file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            event_file.flush()
        finally:
            fcntl.flock(event_file.fileno(), fcntl.LOCK_UN)
