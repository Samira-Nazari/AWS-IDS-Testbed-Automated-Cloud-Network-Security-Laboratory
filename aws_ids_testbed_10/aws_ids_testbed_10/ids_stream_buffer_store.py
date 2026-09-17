"""Persist the continuous IDS row buffer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aws_ids_testbed_10.ids_inference_input import (
    _read_csv,
    _write_csv,
)
from aws_ids_testbed_10.ids_stream_continuity import (
    ContinuityPlan,
    plan_continuous_windows,
)


@dataclass(frozen=True)
class BufferState:
    """Describe the committed stream buffer."""

    path: Path
    row_count: int


def save_waiting_buffer(
    *,
    new_csv_path: Path,
    buffer_output_path: Path,
    existing_buffer_path: Path | None = None,
) -> BufferState:
    """Save rows when fewer than 30 rows are available."""

    new_header, new_rows = _read_csv(Path(new_csv_path))
    existing_rows: list[list[str]] = []

    if existing_buffer_path is not None:
        existing_buffer_path = Path(existing_buffer_path)

        if existing_buffer_path.exists():
            existing_header, existing_rows = _read_csv(
                existing_buffer_path
            )

            if existing_header != new_header:
                raise ValueError(
                    "Buffer and new CSV headers do not match"
                )

    combined_rows = [*existing_rows, *new_rows]
    plan = plan_continuous_windows(len(combined_rows))

    if plan.can_run_inference:
        raise ValueError(
            "Waiting buffer can only be saved when fewer than "
            f"{plan.window_size} rows are available"
        )

    buffer_output_path = Path(buffer_output_path)

    _write_csv(
        output_path=buffer_output_path,
        header=new_header,
        rows=combined_rows,
    )

    return BufferState(
        path=buffer_output_path,
        row_count=len(combined_rows),
    )


def commit_successful_buffer(
    *,
    inference_input_path: Path,
    buffer_output_path: Path,
    plan: ContinuityPlan,
) -> BufferState:
    """Commit retained rows after Step 4 succeeds."""

    if not plan.can_run_inference:
        raise ValueError(
            "Cannot commit a successful inference buffer "
            "without any model windows"
        )

    header, inference_rows = _read_csv(Path(inference_input_path))

    if len(inference_rows) != plan.total_rows:
        raise ValueError(
            "Inference CSV row count does not match continuity plan"
        )

    retained_rows = inference_rows[plan.next_start_offset:]

    if len(retained_rows) != plan.retained_rows:
        raise ValueError(
            "Retained row count does not match continuity plan"
        )

    buffer_output_path = Path(buffer_output_path)

    _write_csv(
        output_path=buffer_output_path,
        header=header,
        rows=retained_rows,
    )

    return BufferState(
        path=buffer_output_path,
        row_count=len(retained_rows),
    )
