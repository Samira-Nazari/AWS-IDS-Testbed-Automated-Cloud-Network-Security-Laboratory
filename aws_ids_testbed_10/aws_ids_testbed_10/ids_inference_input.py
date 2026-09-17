"""Prepare a continuous CSV input for unchanged IDS inference Steps 1–4."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from aws_ids_testbed_10.ids_stream_continuity import (
    ContinuityPlan,
    plan_continuous_windows,
)


@dataclass(frozen=True)
class PreparedInferenceInput:
    """Describe a combined buffer and new-CSV inference input."""

    buffer_rows: int
    new_rows: int
    total_rows: int
    output_path: Path | None
    plan: ContinuityPlan


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read one CSV while preserving its existing row order."""

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)

        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc

        rows = [row for row in reader if row]

    if not header:
        raise ValueError(f"CSV has no header: {path}")

    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"CSV row {row_number} has {len(row)} columns; "
                f"expected {len(header)}: {path}"
            )

    return header, rows


def _write_csv(
    output_path: Path,
    header: list[str],
    rows: list[list[str]],
) -> None:
    """Atomically write a combined inference CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)

    temporary_path.replace(output_path)


def prepare_inference_input(
    *,
    new_csv_path: Path,
    output_path: Path,
    buffer_csv_path: Path | None = None,
) -> PreparedInferenceInput:
    """Combine the committed buffer and one newly received CSV."""

    new_csv_path = Path(new_csv_path)
    output_path = Path(output_path)

    new_header, new_rows = _read_csv(new_csv_path)

    buffer_rows: list[list[str]] = []

    if buffer_csv_path is not None:
        buffer_csv_path = Path(buffer_csv_path)

        if buffer_csv_path.exists():
            buffer_header, buffer_rows = _read_csv(buffer_csv_path)

            if buffer_header != new_header:
                raise ValueError(
                    "Buffer and new CSV headers do not match"
                )

    combined_rows = [*buffer_rows, *new_rows]
    plan = plan_continuous_windows(len(combined_rows))

    prepared_output_path: Path | None = None

    if plan.can_run_inference:
        _write_csv(
            output_path=output_path,
            header=new_header,
            rows=combined_rows,
        )
        prepared_output_path = output_path

    return PreparedInferenceInput(
        buffer_rows=len(buffer_rows),
        new_rows=len(new_rows),
        total_rows=len(combined_rows),
        output_path=prepared_output_path,
        plan=plan,
    )
