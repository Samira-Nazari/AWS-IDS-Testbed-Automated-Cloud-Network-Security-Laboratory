"""
Data Loader module for CIC-DDoS2019 dataset from Kaggle
Handles downloading and loading data from Kaggle
"""

import csv
import os
import subprocess
import pandas as pd
# from TST.TST_Complete_project_V2.config.config import BASE_DIR

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import sys
# from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import BASE_DIR

logger = logging.getLogger(__name__)


def discover_ciciot2023_csv_files(data_dir: Path) -> Dict[str, str]:
    """
    Discover CICIoT2023 CSV files arranged as:
        CSV_files_with_labels/<class_name>/<file>.csv

    Returns a mapping from a stable file key to an absolute CSV path.
    """
    csv_paths = {}

    for csv_file in sorted(data_dir.glob("*/*.csv")):
        class_name = csv_file.parent.name
        file_key = f"{class_name}/{csv_file.stem}"
        csv_paths[file_key] = str(csv_file)

    if not csv_paths:
        raise ValueError(f"No CICIoT2023 CSV files found under: {data_dir}")

    logger.info(f"Discovered {len(csv_paths)} CICIoT2023 CSV files under {data_dir}")
    return csv_paths


def _parse_csv_record(raw_line: bytes) -> list:
    """Parse one CSV record read in binary mode."""
    text = raw_line.decode("utf-8-sig").rstrip("\r\n")
    return next(csv.reader([text]))


def inspect_capture_file(
    file_key: str,
    filepath: str,
    timestamp_column: str,
) -> dict:
    """Collect exact row count and boundary timestamps for one capture CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Capture CSV not found: {path}")

    with path.open("rb") as csv_file:
        header_line = csv_file.readline()
        if not header_line:
            raise ValueError(f"Capture CSV is empty: {path}")

        header = _parse_csv_record(header_line)
        if timestamp_column not in header:
            raise KeyError(
                f"Timestamp column '{timestamp_column}' not found in {path}"
            )
        timestamp_index = header.index(timestamp_column)

        row_count = 0
        first_record = None
        last_record = None

        for raw_line in csv_file:
            if not raw_line.strip():
                continue
            if first_record is None:
                first_record = raw_line
            last_record = raw_line
            row_count += 1

    if row_count == 0 or first_record is None or last_record is None:
        raise ValueError(f"Capture CSV has no data rows: {path}")

    first_values = _parse_csv_record(first_record)
    last_values = _parse_csv_record(last_record)
    expected_columns = len(header)
    if len(first_values) != expected_columns or len(last_values) != expected_columns:
        raise ValueError(
            f"Capture CSV boundary row has an unexpected column count: {path}"
        )

    first_timestamp = float(first_values[timestamp_index])
    last_timestamp = float(last_values[timestamp_index])
    if not np.isfinite(first_timestamp) or not np.isfinite(last_timestamp):
        raise ValueError(f"Capture CSV has a non-finite boundary timestamp: {path}")

    label = file_key.split("/", 1)[0]
    return {
        "Label": label,
        "unique_id": f"Dataset_{file_key}",
        "file_path": str(path.resolve()),
        "file_name": path.name,
        "row_count": row_count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def build_capture_manifest(
    csv_paths: Dict[str, str],
    timestamp_column: str,
) -> pd.DataFrame:
    """Build a chronological, one-row-per-CSV capture inventory."""
    logger.info("\nBuilding CICIoT2023 capture manifest...")
    records = []

    for capture_index, (file_key, filepath) in enumerate(csv_paths.items(), start=1):
        logger.info(
            "  Inspecting capture %d/%d: %s",
            capture_index,
            len(csv_paths),
            file_key,
        )
        records.append(
            inspect_capture_file(
                file_key=file_key,
                filepath=filepath,
                timestamp_column=timestamp_column,
            )
        )

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise ValueError("Capture manifest is empty")
    if manifest["unique_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["unique_id"].duplicated(keep=False), "unique_id"
        ].tolist()
        raise ValueError(f"Duplicate capture unique_id values found: {duplicates}")
    if (manifest["last_timestamp"] < manifest["first_timestamp"]).any():
        invalid_files = manifest.loc[
            manifest["last_timestamp"] < manifest["first_timestamp"],
            "file_path",
        ].tolist()
        raise ValueError(f"Capture timestamp range is reversed: {invalid_files}")

    manifest = manifest.sort_values(
        ["Label", "first_timestamp", "last_timestamp", "file_name"],
        kind="stable",
    ).reset_index(drop=True)
    manifest["chronological_order"] = (
        manifest.groupby("Label", sort=False).cumcount() + 1
    )
    manifest["previous_last_timestamp"] = manifest.groupby(
        "Label", sort=False
    )["last_timestamp"].shift(1)
    manifest["boundary_overlap"] = (
        manifest["previous_last_timestamp"].notna()
        & (
            manifest["first_timestamp"]
            <= manifest["previous_last_timestamp"]
        )
    )

    overlap_count = int(manifest["boundary_overlap"].sum())
    if overlap_count:
        logger.warning(
            "Capture manifest found %d overlapping chronological boundaries",
            overlap_count,
        )
    else:
        logger.info("Capture manifest found no overlapping chronological boundaries")

    logger.info(
        "Capture manifest complete: %d files, %d labels, %d rows",
        len(manifest),
        manifest["Label"].nunique(),
        int(manifest["row_count"].sum()),
    )
    return manifest


def save_capture_manifest(manifest: pd.DataFrame, output_path: Path) -> None:
    """Save capture inventory for later split planning."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    logger.info(f"Saved capture manifest to: {output_path}")


def _calculate_split_targets(
    rows_used: int,
    valid_size: float,
    test_size: float,
) -> dict:
    """Return integer train/validation/test targets that sum to rows_used."""
    train_ratio = 1.0 - valid_size - test_size
    if rows_used <= 0:
        raise ValueError("rows_used must be positive")
    if train_ratio <= 0 or valid_size < 0 or test_size < 0:
        raise ValueError("Invalid train/validation/test proportions")

    train_target = int(rows_used * train_ratio)
    validation_target = int(rows_used * valid_size)
    test_target = rows_used - train_target - validation_target
    return {
        "train": train_target,
        "validation": validation_target,
        "test": test_target,
    }


def _plan_record(
    capture: pd.Series,
    split: str,
    source_start_row: int,
    source_end_row: int,
    class_stats: dict,
) -> dict:
    """Create one half-open source-row segment for the split plan."""
    segment_row_count = source_end_row - source_start_row
    if segment_row_count <= 0:
        raise ValueError("Split-plan segments must contain at least one row")

    return {
        "Label": capture["Label"],
        "unique_id": capture["unique_id"],
        "file_path": capture["file_path"],
        "file_name": capture["file_name"],
        "chronological_order": int(capture["chronological_order"]),
        "split": split,
        "source_start_row": int(source_start_row),
        "source_end_row": int(source_end_row),
        "segment_row_count": int(segment_row_count),
        "capture_row_count": int(capture["row_count"]),
        "class_available_rows": int(class_stats["available_rows"]),
        "class_rows_used": int(class_stats["rows_used"]),
        "train_target": int(class_stats["targets"]["train"]),
        "validation_target": int(class_stats["targets"]["validation"]),
        "test_target": int(class_stats["targets"]["test"]),
        "cap_applied": bool(class_stats["cap_applied"]),
    }


def _plan_all_rows_by_cumulative_position(
    class_manifest: pd.DataFrame,
    class_stats: dict,
) -> list:
    """Use every row and cut exact 60/20/20 boundaries across captures."""
    targets = class_stats["targets"]
    split_ranges = [
        ("train", 0, targets["train"]),
        (
            "validation",
            targets["train"],
            targets["train"] + targets["validation"],
        ),
        (
            "test",
            targets["train"] + targets["validation"],
            class_stats["rows_used"],
        ),
    ]

    records = []
    global_start = 0
    for _, capture in class_manifest.iterrows():
        global_end = global_start + int(capture["row_count"])
        for split, split_start, split_end in split_ranges:
            intersection_start = max(global_start, split_start)
            intersection_end = min(global_end, split_end)
            if intersection_start < intersection_end:
                records.append(
                    _plan_record(
                        capture,
                        split=split,
                        source_start_row=intersection_start - global_start,
                        source_end_row=intersection_end - global_start,
                        class_stats=class_stats,
                    )
                )
        global_start = global_end

    return records


def _plan_capped_capture_pools(
    class_manifest: pd.DataFrame,
    class_stats: dict,
) -> list:
    """Plan exact capped targets while keeping each selected capture in one split."""
    counts = class_manifest["row_count"].astype(int).to_numpy()
    targets = class_stats["targets"]

    prefix_rows = 0
    train_end = None
    for index, count in enumerate(counts):
        prefix_rows += int(count)
        if prefix_rows >= targets["train"]:
            train_end = index + 1
            break

    suffix_rows = 0
    test_start = None
    for index in range(len(counts) - 1, -1, -1):
        suffix_rows += int(counts[index])
        if suffix_rows >= targets["test"]:
            test_start = index
            break

    if train_end is None or test_start is None or train_end > test_start:
        raise ValueError(
            f"Class {class_stats['Label']} cannot preserve capture-exclusive "
            "capped train/test pools with the requested proportions"
        )

    middle_rows = int(counts[train_end:test_start].sum())
    if middle_rows < targets["validation"]:
        raise ValueError(
            f"Class {class_stats['Label']} has only {middle_rows} middle-capture "
            f"rows for a validation target of {targets['validation']}"
        )

    records = []

    remaining = targets["train"]
    for _, capture in class_manifest.iloc[:train_end].iterrows():
        take = min(int(capture["row_count"]), remaining)
        if take > 0:
            records.append(
                _plan_record(capture, "train", 0, take, class_stats)
            )
            remaining -= take
        if remaining == 0:
            break

    remaining = targets["validation"]
    for _, capture in class_manifest.iloc[train_end:test_start].iterrows():
        take = min(int(capture["row_count"]), remaining)
        if take > 0:
            records.append(
                _plan_record(capture, "validation", 0, take, class_stats)
            )
            remaining -= take
        if remaining == 0:
            break

    remaining = targets["test"]
    test_records = []
    for _, capture in class_manifest.iloc[test_start:].iloc[::-1].iterrows():
        take = min(int(capture["row_count"]), remaining)
        if take > 0:
            capture_rows = int(capture["row_count"])
            test_records.append(
                _plan_record(
                    capture,
                    "test",
                    capture_rows - take,
                    capture_rows,
                    class_stats,
                )
            )
            remaining -= take
        if remaining == 0:
            break
    records.extend(reversed(test_records))

    return records


def _records_from_global_split_ranges(
    class_manifest: pd.DataFrame,
    class_stats: dict,
    split_ranges: list,
) -> list:
    """Convert global half-open row ranges into capture-local split records."""
    records = []
    global_start = 0

    for _, capture in class_manifest.iterrows():
        capture_rows = int(capture["row_count"])
        global_end = global_start + capture_rows

        for split, split_start, split_end in split_ranges:
            intersection_start = max(global_start, int(split_start))
            intersection_end = min(global_end, int(split_end))
            if intersection_start < intersection_end:
                records.append(
                    _plan_record(
                        capture,
                        split=split,
                        source_start_row=intersection_start - global_start,
                        source_end_row=intersection_end - global_start,
                        class_stats=class_stats,
                    )
                )

        global_start = global_end

    return records


def _window_count_for_rows(
    row_count: int,
    window_size: int,
    step_size: int,
) -> int:
    """Return how many sliding windows can be created from one row segment."""
    row_count = int(row_count)
    if row_count < window_size:
        return 0
    return ((row_count - window_size) // step_size) + 1


def _rows_for_window_count(
    window_count: int,
    window_size: int,
    step_size: int,
) -> int:
    """Return the minimum consecutive rows needed for window_count windows."""
    window_count = int(window_count)
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    return ((window_count - 1) * step_size) + window_size


def _get_active_window_config() -> tuple:
    """Return the single active window/step configuration used by V9."""
    from config.config import WINDOW_SIZES, STEP_SIZES

    if len(WINDOW_SIZES) != 1 or len(STEP_SIZES) != 1:
        raise ValueError(
            "chronological_distributed_windows requires exactly one active "
            "WINDOW_SIZES value and one active STEP_SIZES value"
        )

    window_size = int(WINDOW_SIZES[0])
    step_size = int(STEP_SIZES[0])
    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window size and step size must be positive")
    if step_size > window_size:
        raise ValueError("Step size cannot be larger than window size")

    return window_size, step_size


def _evenly_spaced_indices(total_count: int, selected_count: int) -> list:
    """Choose selected_count indices spread over [0, total_count)."""
    total_count = int(total_count)
    selected_count = int(selected_count)
    if selected_count <= 0:
        return []
    if selected_count >= total_count:
        return list(range(total_count))
    if selected_count == 1:
        return [total_count // 2]

    raw_indices = np.linspace(0, total_count - 1, selected_count)
    selected = []
    used = set()

    for raw_index in raw_indices:
        candidate = int(round(float(raw_index)))
        if candidate in used:
            offsets = range(1, total_count)
            for offset in offsets:
                left = candidate - offset
                right = candidate + offset
                if 0 <= left < total_count and left not in used:
                    candidate = left
                    break
                if 0 <= right < total_count and right not in used:
                    candidate = right
                    break
        if candidate not in used:
            selected.append(candidate)
            used.add(candidate)

    if len(selected) < selected_count:
        for candidate in range(total_count):
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
                if len(selected) == selected_count:
                    break

    return sorted(selected)


def _choose_file_exclusive_window_regions(
    class_manifest: pd.DataFrame,
    window_targets: dict,
    window_size: int,
    step_size: int,
) -> dict | None:
    """
    Split chronological captures into train/validation/test file regions.

    The chosen regions are file-exclusive and must each have enough possible
    windows for the requested split target. Among feasible choices, prefer
    boundaries close to the 60/20/20 chronological window proportions.
    """
    if len(class_manifest) < 3:
        return None

    file_windows = np.array(
        [
            _window_count_for_rows(row_count, window_size, step_size)
            for row_count in class_manifest["row_count"].astype(int)
        ],
        dtype=np.int64,
    )
    total_windows = int(file_windows.sum())
    target_total = int(sum(window_targets.values()))
    if total_windows < target_total:
        return None

    cumulative_windows = np.concatenate([[0], np.cumsum(file_windows)])
    target_train_ratio = window_targets["train"] / target_total
    target_val_ratio = window_targets["validation"] / target_total
    ideal_train_end = total_windows * target_train_ratio
    ideal_val_end = total_windows * (target_train_ratio + target_val_ratio)

    best_choice = None
    best_score = None
    num_files = len(class_manifest)

    for train_end in range(1, num_files - 1):
        train_windows = int(cumulative_windows[train_end])
        if train_windows < window_targets["train"]:
            continue

        for validation_end in range(train_end + 1, num_files):
            validation_windows = int(
                cumulative_windows[validation_end] - cumulative_windows[train_end]
            )
            test_windows = int(total_windows - cumulative_windows[validation_end])

            if validation_windows < window_targets["validation"]:
                continue
            if test_windows < window_targets["test"]:
                continue

            score = (
                abs(train_windows - ideal_train_end)
                + abs((train_windows + validation_windows) - ideal_val_end)
            )

            if best_score is None or score < best_score:
                best_score = score
                best_choice = (train_end, validation_end)

    if best_choice is None:
        return None

    train_end, validation_end = best_choice
    return {
        "train": class_manifest.iloc[:train_end].copy(),
        "validation": class_manifest.iloc[train_end:validation_end].copy(),
        "test": class_manifest.iloc[validation_end:].copy(),
    }


def _build_window_chunk_candidates(
    region_manifest: pd.DataFrame,
    window_size: int,
    step_size: int,
    chunk_windows: int,
) -> list:
    """Build chronological candidate chunks of complete sliding windows."""
    candidates = []
    for _, capture in region_manifest.iterrows():
        capture_windows = _window_count_for_rows(
            int(capture["row_count"]),
            window_size,
            step_size,
        )
        if capture_windows <= 0:
            continue

        for window_start_index in range(0, capture_windows, chunk_windows):
            window_count = min(chunk_windows, capture_windows - window_start_index)
            source_start_row = window_start_index * step_size
            source_end_row = source_start_row + _rows_for_window_count(
                window_count,
                window_size,
                step_size,
            )
            candidates.append(
                {
                    "capture": capture,
                    "window_start_index": int(window_start_index),
                    "window_count": int(window_count),
                    "source_start_row": int(source_start_row),
                    "source_end_row": int(source_end_row),
                }
            )

    return candidates


def _merge_window_chunks_to_records(
    selected_chunks: list,
    split: str,
    class_stats: dict,
    window_size: int,
    step_size: int,
) -> list:
    """Merge adjacent/overlapping selected chunks into row-plan records."""
    if not selected_chunks:
        return []

    selected_chunks = sorted(
        selected_chunks,
        key=lambda chunk: (
            int(chunk["capture"]["chronological_order"]),
            int(chunk["source_start_row"]),
        ),
    )

    merged = []
    current = None

    for chunk in selected_chunks:
        capture = chunk["capture"]
        unique_id = capture["unique_id"]
        source_start_row = int(chunk["source_start_row"])
        source_end_row = int(chunk["source_end_row"])
        window_count = int(chunk["window_count"])

        if (
            current is not None
            and current["unique_id"] == unique_id
            and source_start_row <= current["source_end_row"]
        ):
            current["source_end_row"] = max(
                current["source_end_row"],
                source_end_row,
            )
            current["window_count"] += window_count
            continue

        if current is not None:
            merged.append(current)

        current = {
            "capture": capture,
            "unique_id": unique_id,
            "source_start_row": source_start_row,
            "source_end_row": source_end_row,
            "window_count": window_count,
        }

    if current is not None:
        merged.append(current)

    records = []
    for chunk in merged:
        segment_rows = int(chunk["source_end_row"] - chunk["source_start_row"])
        actual_windows = _window_count_for_rows(
            segment_rows,
            window_size,
            step_size,
        )
        if actual_windows != int(chunk["window_count"]):
            raise ValueError(
                "Merged window chunk produced an unexpected window count: "
                f"expected {chunk['window_count']}, got {actual_windows}"
            )

        records.append(
            _plan_record(
                chunk["capture"],
                split=split,
                source_start_row=int(chunk["source_start_row"]),
                source_end_row=int(chunk["source_end_row"]),
                class_stats=class_stats,
            )
        )

    return records


def _select_distributed_window_records(
    region_manifest: pd.DataFrame,
    split: str,
    target_windows: int,
    class_stats: dict,
    window_size: int,
    step_size: int,
    chunk_windows: int,
) -> list:
    """Select complete window chunks evenly distributed over a split region."""
    target_windows = int(target_windows)
    if target_windows <= 0:
        return []

    candidates = _build_window_chunk_candidates(
        region_manifest,
        window_size,
        step_size,
        chunk_windows,
    )
    total_candidate_windows = int(
        sum(candidate["window_count"] for candidate in candidates)
    )
    if total_candidate_windows < target_windows:
        raise ValueError(
            f"Class {class_stats['Label']} split {split} has only "
            f"{total_candidate_windows} candidate windows for target "
            f"{target_windows}"
        )

    if target_windows == total_candidate_windows:
        selected_chunks = candidates
    else:
        chunks_needed = int(np.ceil(target_windows / chunk_windows))
        selected_indices = _evenly_spaced_indices(len(candidates), chunks_needed)
        selected_index_set = set(selected_indices)

        while (
            sum(candidates[index]["window_count"] for index in selected_index_set)
            < target_windows
        ):
            remaining_indices = [
                index
                for index in range(len(candidates))
                if index not in selected_index_set
            ]
            if not remaining_indices:
                break
            selected_index_set.add(remaining_indices[len(remaining_indices) // 2])

        selected_chunks = []
        remaining_windows = target_windows
        for index in sorted(selected_index_set):
            if remaining_windows <= 0:
                break
            candidate = candidates[index].copy()
            take_windows = min(int(candidate["window_count"]), remaining_windows)
            candidate["window_count"] = take_windows
            candidate["source_end_row"] = (
                int(candidate["source_start_row"])
                + _rows_for_window_count(take_windows, window_size, step_size)
            )
            selected_chunks.append(candidate)
            remaining_windows -= take_windows

        if remaining_windows > 0:
            for index, candidate in enumerate(candidates):
                if index in selected_index_set:
                    continue
                if remaining_windows <= 0:
                    break
                candidate = candidate.copy()
                take_windows = min(int(candidate["window_count"]), remaining_windows)
                candidate["window_count"] = take_windows
                candidate["source_end_row"] = (
                    int(candidate["source_start_row"])
                    + _rows_for_window_count(take_windows, window_size, step_size)
                )
                selected_chunks.append(candidate)
                remaining_windows -= take_windows

        if remaining_windows > 0:
            raise ValueError(
                f"Could not select {target_windows} distributed windows for "
                f"{class_stats['Label']}/{split}"
            )

    records = _merge_window_chunks_to_records(
        selected_chunks,
        split=split,
        class_stats=class_stats,
        window_size=window_size,
        step_size=step_size,
    )
    actual_windows = sum(
        _window_count_for_rows(
            record["segment_row_count"],
            window_size,
            step_size,
        )
        for record in records
    )
    if actual_windows != target_windows:
        raise ValueError(
            f"Distributed window selection for {class_stats['Label']}/{split} "
            f"expected {target_windows} windows, got {actual_windows}"
        )

    return records


def _count_record_windows_by_split(
    records: list,
    window_size: int,
    step_size: int,
) -> dict:
    """Count generated windows by split for a list of row-plan records."""
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        split = record["split"]
        if split not in counts:
            continue
        counts[split] += _window_count_for_rows(
            int(record["segment_row_count"]),
            window_size,
            step_size,
        )
    return counts


def _plan_chronological_distributed_window_pools(
    class_manifest: pd.DataFrame,
    class_stats: dict,
) -> list:
    """
    Plan split records by selecting complete windows distributed over each region.

    Preferred behavior uses file-exclusive chronological regions:
      - early files -> train region;
      - middle files -> validation region;
      - later files -> test region.

    Inside each region, complete sliding-window chunks are selected evenly across
    the available chronological window starts. Samples inside each selected
    window remain consecutive. If file-exclusive regions cannot satisfy the
    window targets, this falls back to row-range chronological splitting.
    """
    from config.config import (
        CAPTURE_SPLIT_PREFER_FILE_EXCLUSIVE,
        TEST_SIZE,
        VALID_SIZE,
        WINDOW_SPLIT_CHUNK_WINDOWS,
        WINDOW_SPLIT_SELECTION,
    )

    if WINDOW_SPLIT_SELECTION != "evenly_spaced":
        raise ValueError(
            "Unsupported WINDOW_SPLIT_SELECTION="
            f"{WINDOW_SPLIT_SELECTION!r}; expected 'evenly_spaced'"
        )

    window_size, step_size = _get_active_window_config()
    chunk_windows = int(WINDOW_SPLIT_CHUNK_WINDOWS)
    if chunk_windows <= 0:
        raise ValueError("WINDOW_SPLIT_CHUNK_WINDOWS must be positive")

    total_possible_windows = int(
        sum(
            _window_count_for_rows(row_count, window_size, step_size)
            for row_count in class_manifest["row_count"].astype(int)
        )
    )
    requested_windows = min(
        _window_count_for_rows(
            int(class_stats["rows_used"]),
            window_size,
            step_size,
        ),
        total_possible_windows,
    )
    if requested_windows <= 0:
        raise ValueError(
            f"Class {class_stats['Label']} has no usable windows for "
            f"window_size={window_size}, step_size={step_size}"
        )

    window_targets = _calculate_split_targets(
        requested_windows,
        valid_size=VALID_SIZE,
        test_size=TEST_SIZE,
    )
    class_stats["window_size"] = window_size
    class_stats["step_size"] = step_size
    class_stats["window_targets"] = window_targets
    class_stats["requested_windows"] = requested_windows
    class_stats["total_possible_windows"] = total_possible_windows

    if CAPTURE_SPLIT_PREFER_FILE_EXCLUSIVE:
        regions = _choose_file_exclusive_window_regions(
            class_manifest,
            window_targets,
            window_size,
            step_size,
        )
        if regions is not None:
            records = []
            for split in ["train", "validation", "test"]:
                records.extend(
                    _select_distributed_window_records(
                        regions[split],
                        split=split,
                        target_windows=window_targets[split],
                        class_stats=class_stats,
                        window_size=window_size,
                        step_size=step_size,
                        chunk_windows=chunk_windows,
                    )
                )

            class_stats["window_split_strategy"] = (
                "file_exclusive_distributed_windows"
            )
            logger.info(
                "Chronological distributed-window split %s: "
                "file-exclusive regions, target windows train=%d validation=%d test=%d",
                class_stats["Label"],
                window_targets["train"],
                window_targets["validation"],
                window_targets["test"],
            )
            return records

    logger.info(
        "Chronological distributed-window split %s: file-exclusive window "
        "regions could not satisfy targets; falling back to row-range split",
        class_stats["Label"],
    )
    records = _plan_chronological_block_capture_pools(
        class_manifest,
        class_stats,
    )
    fallback_window_targets = _count_record_windows_by_split(
        records,
        window_size,
        step_size,
    )
    class_stats["window_targets"] = fallback_window_targets
    class_stats["window_split_strategy"] = "row_range_fallback"
    return records


def _plan_chronological_block_capture_pools(
    class_manifest: pd.DataFrame,
    class_stats: dict,
) -> list:
    """
    Plan chronological train/validation/test blocks for one class.

    Preferred behavior:
      - train uses earlier capture files;
      - validation uses middle capture files;
      - test uses later capture files;
      - selected files are exclusive to one split whenever possible.

    Fallback behavior:
      - when whole-file exclusive assignment cannot satisfy the requested
        60/20/20 targets, use chronological row ranges;
      - keep rows disjoint;
      - insert an unused boundary gap when available.
    """
    from config.config import (
        CAPTURE_SPLIT_BOUNDARY_GAP_ROWS,
        CAPTURE_SPLIT_PREFER_FILE_EXCLUSIVE,
    )

    targets = class_stats["targets"]
    split_order = ["train", "validation", "test"]
    class_label = class_stats["Label"]
    total_available = int(class_manifest["row_count"].astype(int).sum())

    def try_file_exclusive_blocks() -> list | None:
        records = []
        capture_index = 0
        num_captures = len(class_manifest)

        for split in split_order:
            remaining = int(targets[split])
            while remaining > 0 and capture_index < num_captures:
                capture = class_manifest.iloc[capture_index]
                capture_rows = int(capture["row_count"])
                take = min(capture_rows, remaining)

                if take > 0:
                    records.append(
                        _plan_record(
                            capture,
                            split=split,
                            source_start_row=0,
                            source_end_row=take,
                            class_stats=class_stats,
                        )
                    )
                    remaining -= take

                # Preserve capture exclusivity: if this split only needs part
                # of the current file, leave the unused remainder out of all
                # splits and start the next split at the next capture file.
                capture_index += 1

            if remaining > 0:
                return None

        return records

    if CAPTURE_SPLIT_PREFER_FILE_EXCLUSIVE:
        exclusive_records = try_file_exclusive_blocks()
        if exclusive_records is not None:
            logger.info(
                "Chronological-block split %s: used file-exclusive split assignment",
                class_label,
            )
            return exclusive_records

        logger.info(
            "Chronological-block split %s: file-exclusive assignment could not "
            "satisfy targets; falling back to row-range boundaries",
            class_label,
        )

    configured_gap = max(0, int(CAPTURE_SPLIT_BOUNDARY_GAP_ROWS))
    requested_rows = int(class_stats["rows_used"])
    available_gap_budget = max(0, total_available - requested_rows)
    effective_gap = min(configured_gap, available_gap_budget // 2)

    if configured_gap and effective_gap < configured_gap:
        logger.info(
            "Chronological-block split %s: reduced boundary gap from %d to %d "
            "rows because only %d unused rows are available",
            class_label,
            configured_gap,
            effective_gap,
            available_gap_budget,
        )

    train_start = 0
    train_end = train_start + int(targets["train"])
    validation_start = train_end + effective_gap
    validation_end = validation_start + int(targets["validation"])
    test_start = validation_end + effective_gap
    test_end = test_start + int(targets["test"])

    if test_end > total_available:
        raise ValueError(
            f"Chronological-block split for {class_label} requires {test_end} "
            f"rows including boundary gaps, but only {total_available} are available"
        )

    split_ranges = [
        ("train", train_start, train_end),
        ("validation", validation_start, validation_end),
        ("test", test_start, test_end),
    ]
    logger.info(
        "Chronological-block split %s: using row-range boundaries with %d-row gaps",
        class_label,
        effective_gap,
    )
    return _records_from_global_split_ranges(
        class_manifest,
        class_stats,
        split_ranges,
    )


def _add_unused_plan_segments(
    selected_records: list,
    manifest: pd.DataFrame,
    class_stats_by_label: dict,
) -> list:
    """Add the complement of selected source ranges as explicit unused segments."""
    all_records = list(selected_records)
    selected_by_capture = {}
    for record in selected_records:
        selected_by_capture.setdefault(record["unique_id"], []).append(
            (record["source_start_row"], record["source_end_row"])
        )

    for _, capture in manifest.iterrows():
        pointer = 0
        capture_rows = int(capture["row_count"])
        intervals = sorted(selected_by_capture.get(capture["unique_id"], []))
        class_stats = class_stats_by_label[capture["Label"]]

        for start, end in intervals:
            if start < pointer:
                raise ValueError(
                    f"Overlapping split-plan ranges for {capture['unique_id']}"
                )
            if pointer < start:
                all_records.append(
                    _plan_record(
                        capture,
                        "unused",
                        pointer,
                        start,
                        class_stats,
                    )
                )
            pointer = end

        if pointer < capture_rows:
            all_records.append(
                _plan_record(
                    capture,
                    "unused",
                    pointer,
                    capture_rows,
                    class_stats,
                )
            )

    return all_records


def build_capture_split_plan(
    manifest: pd.DataFrame,
    max_samples_per_label: int,
    valid_size: float,
    test_size: float,
) -> pd.DataFrame:
    """Plan exact capped row allocations without changing loaded data yet."""
    from config.config import CAPTURE_SPLIT_POLICY

    split_policy = CAPTURE_SPLIT_POLICY
    valid_split_policies = {
        "legacy_capture_pools",
        "chronological_block",
        "chronological_distributed_windows",
    }
    if split_policy not in valid_split_policies:
        raise ValueError(
            f"Unsupported CAPTURE_SPLIT_POLICY={split_policy!r}. "
            f"Expected one of: {sorted(valid_split_policies)}"
        )

    logger.info("Capture split policy: %s", split_policy)

    required_columns = {
        "Label",
        "unique_id",
        "file_path",
        "file_name",
        "row_count",
        "chronological_order",
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Capture manifest is missing required columns: {sorted(missing_columns)}"
        )

    selected_records = []
    class_stats_by_label = {}

    for label, class_manifest in manifest.groupby("Label", sort=False):
        class_manifest = class_manifest.sort_values(
            "chronological_order", kind="stable"
        ).reset_index(drop=True)
        available_rows = int(class_manifest["row_count"].sum())
        if max_samples_per_label and max_samples_per_label > 0:
            rows_used = min(available_rows, int(max_samples_per_label))
        else:
            rows_used = available_rows

        class_stats = {
            "Label": label,
            "available_rows": available_rows,
            "rows_used": rows_used,
            "cap_applied": rows_used < available_rows,
            "valid_size": valid_size,
            "test_size": test_size,
            "targets": _calculate_split_targets(
                rows_used,
                valid_size=valid_size,
                test_size=test_size,
            ),
        }
        class_stats_by_label[label] = class_stats

        if split_policy == "chronological_distributed_windows":
            records = _plan_chronological_distributed_window_pools(
                class_manifest,
                class_stats,
            )
        elif split_policy == "chronological_block":
            records = _plan_chronological_block_capture_pools(
                class_manifest,
                class_stats,
            )
        elif class_stats["cap_applied"]:
            records = _plan_capped_capture_pools(class_manifest, class_stats)
        else:
            records = _plan_all_rows_by_cumulative_position(
                class_manifest,
                class_stats,
            )
        selected_records.extend(records)

        logger.info(
            "Split plan %s: available=%d used=%d train=%d validation=%d test=%d cap=%s",
            label,
            available_rows,
            rows_used,
            class_stats["targets"]["train"],
            class_stats["targets"]["validation"],
            class_stats["targets"]["test"],
            class_stats["cap_applied"],
        )
        if split_policy == "chronological_distributed_windows":
            logger.info(
                "Split plan %s windows: possible=%d requested=%d train=%d "
                "validation=%d test=%d strategy=%s",
                label,
                class_stats["total_possible_windows"],
                class_stats["requested_windows"],
                class_stats["window_targets"]["train"],
                class_stats["window_targets"]["validation"],
                class_stats["window_targets"]["test"],
                class_stats["window_split_strategy"],
            )

    all_records = _add_unused_plan_segments(
        selected_records,
        manifest,
        class_stats_by_label,
    )
    plan = pd.DataFrame.from_records(all_records)
    split_rank = {"train": 0, "validation": 1, "test": 2, "unused": 3}
    plan["_split_rank"] = plan["split"].map(split_rank)
    plan = plan.sort_values(
        [
            "Label",
            "chronological_order",
            "source_start_row",
            "_split_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)
    plan = plan.drop(columns="_split_rank")
    plan["segment_order"] = plan.groupby("Label", sort=False).cumcount() + 1
    plan["is_partial_capture"] = (
        plan["segment_row_count"] < plan["capture_row_count"]
    )
    if split_policy == "chronological_distributed_windows":
        window_size, step_size = _get_active_window_config()
        plan["segment_window_count"] = plan["segment_row_count"].map(
            lambda row_count: _window_count_for_rows(
                int(row_count),
                window_size,
                step_size,
            )
        )

    selected_plan = plan[plan["split"] != "unused"]
    for label, class_stats in class_stats_by_label.items():
        label_plan = selected_plan[selected_plan["Label"] == label]
        if split_policy == "chronological_distributed_windows":
            actual_counts = label_plan.groupby("split")[
                "segment_window_count"
            ].sum()
            for split, expected_count in class_stats["window_targets"].items():
                actual_count = int(actual_counts.get(split, 0))
                if actual_count != expected_count:
                    raise ValueError(
                        f"Split-plan window-count mismatch for {label}/{split}: "
                        f"expected {expected_count}, got {actual_count}"
                    )
        else:
            actual_counts = label_plan.groupby("split")["segment_row_count"].sum()
            for split, expected_count in class_stats["targets"].items():
                actual_count = int(actual_counts.get(split, 0))
                if actual_count != expected_count:
                    raise ValueError(
                        f"Split-plan count mismatch for {label}/{split}: "
                        f"expected {expected_count}, got {actual_count}"
                    )

        if class_stats["cap_applied"] and split_policy == "legacy_capture_pools":
            selected_split_counts = label_plan.groupby("unique_id")["split"].nunique()
            if (selected_split_counts > 1).any():
                raise ValueError(
                    f"Capped class {label} reuses a capture across data splits"
                )

    coverage = plan.groupby("unique_id")["segment_row_count"].sum()
    expected_coverage = manifest.set_index("unique_id")["row_count"].astype(int)
    if not coverage.sort_index().equals(expected_coverage.sort_index()):
        raise ValueError("Split plan does not account for every source row")

    logger.info(
        "Capture split plan complete: %d selected rows, %d unused rows",
        int(selected_plan["segment_row_count"].sum()),
        int(plan.loc[plan["split"] == "unused", "segment_row_count"].sum()),
    )
    return plan


def save_capture_split_plan(plan: pd.DataFrame, output_path: Path) -> None:
    """Save the capture-aware allocation plan for later loading/windowing steps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_path, index=False)
    logger.info(f"Saved capture split plan to: {output_path}")


def cap_label_by_random_chronological_block(
    df: pd.DataFrame,
    label_column: str,
    timestamp_column: str,
    max_samples_per_label: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Keep one consecutive chronological block per label.

    This avoids random row sampling, which can break temporal continuity before
    sliding-window creation.
    """
    capped_parts = []
    rng = np.random.default_rng(random_state)

    for label, label_df in df.groupby(label_column, sort=False):
        label_df = label_df.sort_values(timestamp_column).reset_index(drop=True)
        label_count = len(label_df)

        if label_count <= max_samples_per_label:
            capped_parts.append(label_df)
            logger.info(f"    - {label}: kept all {label_count} rows")
            continue

        max_start = label_count - max_samples_per_label
        start_idx = int(rng.integers(0, max_start + 1))
        end_idx = start_idx + max_samples_per_label

        selected_df = label_df.iloc[start_idx:end_idx].copy()
        capped_parts.append(selected_df)

        logger.info(
            f"    - {label}: kept {len(selected_df)} consecutive rows "
            f"from sorted positions {start_idx} to {end_idx - 1}; "
            f"dropped {label_count - len(selected_df)}"
        )

    return (
        pd.concat(capped_parts, ignore_index=True)
        .sort_values([label_column, timestamp_column])
        .reset_index(drop=True)
    )


class KaggleDataLoader:
    """Load CIC-DDoS2019 dataset from Kaggle"""
    
    def __init__(self, kaggle_dataset_name: str, download_path: str):
        """
        Initialize Kaggle data loader
        
        Args:
            kaggle_dataset_name: Kaggle dataset name (e.g., 'rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files')
            download_path: Path to download dataset to
        """
        self.dataset_name = kaggle_dataset_name
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are configured"""
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        return kaggle_config.exists()
    
    def download_dataset(self) -> bool:
        """
        Download dataset from Kaggle using kaggle-cli
        
        Returns:
            True if successful, False otherwise
        """
        if not self.check_kaggle_credentials():
            logger.error("❌ Kaggle credentials not found at ~/.kaggle/kaggle.json")
            logger.error("   Please set up Kaggle API credentials:")
            logger.error("   1. Go to https://www.kaggle.com/settings/account")
            logger.error("   2. Click 'Create New API Token' to download kaggle.json")
            logger.error("   3. Save it to ~/.kaggle/kaggle.json")
            logger.error("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        logger.info(f"📥 Downloading dataset from Kaggle: {self.dataset_name}")
        
        try:
            # Download dataset using kaggle CLI
            cmd = [
                'kaggle', 'datasets', 'download',
                '-d', self.dataset_name,
                '-p', str(self.download_path),
                '--unzip'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✓ Dataset downloaded successfully to {self.download_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            return False
        except FileNotFoundError:
            logger.error("❌ kaggle CLI not found. Install with: pip install kaggle")
            return False


class DataLoader:
    """Load and parse CIC-DDoS2019 data from CSV files"""
    
    def __init__(
        self,
        csv_paths: Dict[str, str],
        timestamp_format: str = '%Y-%m-%d %H:%M:%S',
        split_plan: pd.DataFrame = None,
    ):
        """
        Initialize data loader
        
        Args:
            csv_paths: Dictionary mapping protocol names to CSV file paths
            timestamp_format: Format of timestamp column
        """
        self.csv_paths = csv_paths
        self.timestamp_format = timestamp_format
        self.split_plan = split_plan
        
    def to_unix_timestamp(self, date_string: str) -> int:
        """Convert timestamp string to Unix timestamp"""
        try:
            timestamp_str = str(date_string).split('.')[0]  # Remove milliseconds
            return int(time.mktime(time.strptime(timestamp_str, self.timestamp_format)))
        except (ValueError, TypeError):
            return None
    
    def preprocess_csv(
        self,
        filepath: str,
        file_key: str,
        unique_id_prefix: str,
        balance_classes: bool = False,
        max_rows: int = None,
        source_start_row: int = 0,
        source_end_row: int = None,
        source_unique_id: str = None,
        planned_split: str = None,
        split_segment_id: str = None,
    ) -> pd.DataFrame:
        """
        Read and preprocess a single CSV file
        
        Args:
            filepath: Path to CSV file
            file_key: Config key identifying the source CSV file
            unique_id_prefix: Prefix for unique ID column
            balance_classes: If True, downsample each non-BENIGN label to the
                BENIGN count within this file. Disabled by default.
            max_rows: Optional maximum rows to read from this CSV.
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"📖 Loading CSV: {filepath}")

        from config.config import (
            BENIGN_LABEL,
            SOURCE_LABEL_COLUMN,
            SOURCE_TIMESTAMP_COLUMN,
            TARGET_LABEL_COLUMN,
            TARGET_TIMESTAMP_COLUMN,
        )
        
        read_kwargs = {"low_memory": False}
        if source_start_row < 0:
            raise ValueError("source_start_row must be non-negative")
        if source_end_row is not None:
            if max_rows is not None:
                raise ValueError(
                    "max_rows and source_end_row cannot be used together"
                )
            if source_end_row <= source_start_row:
                raise ValueError(
                    "source_end_row must be greater than source_start_row"
                )
            read_kwargs["nrows"] = source_end_row - source_start_row
            if source_start_row:
                read_kwargs["skiprows"] = range(1, source_start_row + 1)
            logger.info(
                "   Reading planned source rows [%d, %d)",
                source_start_row,
                source_end_row,
            )
        elif max_rows is not None:
            read_kwargs["nrows"] = max_rows
            logger.info(f"   Reading at most {max_rows} rows for cap control")

        df = pd.read_csv(filepath, **read_kwargs)
        df.columns = df.columns.str.strip()
        df["source_row_index"] = np.arange(
            source_start_row,
            source_start_row + len(df),
            dtype=np.int64,
        )
        
        logger.info(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        
        if SOURCE_LABEL_COLUMN not in df.columns:
            raise KeyError(
                f"Label column '{SOURCE_LABEL_COLUMN}' not found in {filepath}"
            )

        if SOURCE_TIMESTAMP_COLUMN not in df.columns:
            raise KeyError(
                f"Timestamp column '{SOURCE_TIMESTAMP_COLUMN}' not found in {filepath}"
            )

        rename_columns = {}
        if SOURCE_LABEL_COLUMN != TARGET_LABEL_COLUMN:
            rename_columns[SOURCE_LABEL_COLUMN] = TARGET_LABEL_COLUMN
        if SOURCE_TIMESTAMP_COLUMN != TARGET_TIMESTAMP_COLUMN:
            rename_columns[SOURCE_TIMESTAMP_COLUMN] = TARGET_TIMESTAMP_COLUMN
        if rename_columns:
            df = df.rename(columns=rename_columns)

        df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].astype(str).str.strip()
        df[TARGET_TIMESTAMP_COLUMN] = pd.to_numeric(
            df[TARGET_TIMESTAMP_COLUMN],
            errors="coerce",
        )

        combined = df[
            df[TARGET_LABEL_COLUMN].notna()
            & (df[TARGET_LABEL_COLUMN] != "")
            & df[TARGET_TIMESTAMP_COLUMN].notna()
        ].copy()

        logger.info("   Keeping labels exactly as they appear in the CSV rows")
        logger.info("   Label distribution in this file:")
        for label, count in combined[TARGET_LABEL_COLUMN].value_counts().items():
            logger.info(f"     - {label}: {count}")

        if balance_classes:
            benign_df = combined[combined[TARGET_LABEL_COLUMN] == BENIGN_LABEL].copy()
            balanced_parts = [benign_df]
            benign_count = len(benign_df)

            for label, label_df in combined[
                combined[TARGET_LABEL_COLUMN] != BENIGN_LABEL
            ].groupby(TARGET_LABEL_COLUMN):
                if benign_count and len(label_df) > benign_count:
                    label_df = label_df.sample(n=benign_count, random_state=42)
                    logger.info(f"   - Balanced {label} to {len(label_df)} samples")
                balanced_parts.append(label_df)

            combined = pd.concat(balanced_parts, ignore_index=True)
        
        # Add unique identifier
        combined['unique_id'] = (
            source_unique_id
            if source_unique_id is not None
            else f'{unique_id_prefix}_{file_key}'
        )
        if planned_split is not None:
            combined["planned_split"] = planned_split
        if split_segment_id is not None:
            combined["split_segment_id"] = split_segment_id
        
        logger.info(f"   Valid samples: {len(combined)}")
        
        return combined

    def _load_planned_data(self, balance_classes: bool = False) -> pd.DataFrame:
        """Load only selected source-row segments from the capture split plan."""
        plan = self.split_plan.copy()
        selected_plan = plan[plan["split"] != "unused"].copy()
        if selected_plan.empty:
            raise ValueError("Capture split plan has no selected rows")

        selected_plan = selected_plan.sort_values(
            ["Label", "chronological_order", "source_start_row"],
            kind="stable",
        ).reset_index(drop=True)

        combined_parts = []
        for plan_index, segment in selected_plan.iterrows():
            file_key = str(segment["unique_id"])
            if file_key.startswith("Dataset_"):
                file_key = file_key[len("Dataset_"):]

            split_segment_id = (
                f"{segment['unique_id']}::{segment['split']}::"
                f"{int(segment['segment_order'])}"
            )
            logger.info(
                "Loading planned segment %d/%d: %s %s rows [%d, %d)",
                plan_index + 1,
                len(selected_plan),
                segment["unique_id"],
                segment["split"],
                int(segment["source_start_row"]),
                int(segment["source_end_row"]),
            )

            segment_df = self.preprocess_csv(
                filepath=str(segment["file_path"]),
                file_key=file_key,
                unique_id_prefix="Dataset",
                balance_classes=balance_classes,
                source_start_row=int(segment["source_start_row"]),
                source_end_row=int(segment["source_end_row"]),
                source_unique_id=str(segment["unique_id"]),
                planned_split=str(segment["split"]),
                split_segment_id=split_segment_id,
            )

            expected_rows = int(segment["segment_row_count"])
            if len(segment_df) != expected_rows:
                raise ValueError(
                    f"Planned segment {split_segment_id} expected {expected_rows} "
                    f"valid rows but loaded {len(segment_df)}"
                )
            combined_parts.append(segment_df)

        combined_df = pd.concat(combined_parts, ignore_index=True)

        duplicate_rows = combined_df.duplicated(
            subset=["unique_id", "source_row_index"],
            keep=False,
        )
        if duplicate_rows.any():
            examples = combined_df.loc[
                duplicate_rows,
                ["unique_id", "source_row_index", "planned_split"],
            ].head(10)
            raise ValueError(
                "Capture split plan loaded duplicate source rows:\n"
                f"{examples.to_string(index=False)}"
            )

        expected_counts = (
            selected_plan.groupby(["Label", "split"])["segment_row_count"]
            .sum()
            .sort_index()
        )
        actual_counts = (
            combined_df.groupby(["Label", "planned_split"])
            .size()
            .sort_index()
        )
        actual_counts.index = actual_counts.index.set_names(["Label", "split"])
        if not actual_counts.equals(expected_counts.astype(actual_counts.dtype)):
            comparison = pd.concat(
                [
                    expected_counts.rename("expected"),
                    actual_counts.rename("actual"),
                ],
                axis=1,
            ).fillna(0)
            raise ValueError(
                "Loaded split counts do not match the capture plan:\n"
                f"{comparison.to_string()}"
            )

        split_order = pd.CategoricalDtype(
            ["train", "validation", "test"], ordered=True
        )
        combined_df["planned_split"] = combined_df["planned_split"].astype(
            split_order
        )
        combined_df = combined_df.sort_values(
            [
                "planned_split",
                "Label",
                "Timestamp",
                "unique_id",
                "source_row_index",
            ],
            kind="stable",
        ).reset_index(drop=True)
        combined_df["planned_split"] = combined_df["planned_split"].astype(str)

        logger.info("\nCapture-plan loading complete")
        logger.info(f"  Loaded rows: {len(combined_df)}")
        logger.info("  Split counts:")
        for (label, split), count in actual_counts.items():
            logger.info(f"    - {label}/{split}: {int(count)}")

        return combined_df
    
    def load_all_data(self, balance_classes: bool = True) -> pd.DataFrame:
        """
        Load and combine all CSV files
        
        Args:
            balance_classes: If True, balance benign vs attack in combined data
            
        Returns:
            Combined DataFrame with all data
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*60)
        
        combined_dfs = []
        
        from config.config import (
            APPLY_MAX_SAMPLES_PER_LABEL,
            BALANCE_CLASSES,
            DATASET_NAME,
            MAX_SAMPLES_PER_LABEL,
            NORMALIZE_LABELS,
            RANDOM_STATE,
        )

        if self.split_plan is not None:
            logger.info(
                "Using capture split plan; bypassing legacy early-read and "
                "second-stage per-label capping"
            )
            return self._load_planned_data(balance_classes=BALANCE_CLASSES)

        label_counts = {}
        for file_key, filepath in self.csv_paths.items():
            if not os.path.exists(filepath):
                logger.warning(f"⚠️  File not found: {filepath}")
                continue

            max_rows_for_file = None
            if (
                DATASET_NAME == "CICIoT2023"
                and APPLY_MAX_SAMPLES_PER_LABEL
                and MAX_SAMPLES_PER_LABEL
            ):
                expected_label = file_key.split("/", 1)[0]
                remaining_rows = MAX_SAMPLES_PER_LABEL - label_counts.get(
                    expected_label,
                    0,
                )
                if remaining_rows <= 0:
                    logger.info(
                        f"Skipping {filepath}: {expected_label} already reached "
                        f"cap {MAX_SAMPLES_PER_LABEL}"
                    )
                    continue
                max_rows_for_file = remaining_rows
            
            try:
                df = self.preprocess_csv(
                    filepath,
                    file_key,
                    'Dataset',
                    balance_classes=BALANCE_CLASSES,
                    max_rows=max_rows_for_file,
                )

                if APPLY_MAX_SAMPLES_PER_LABEL and MAX_SAMPLES_PER_LABEL:
                    capped_parts = []
                    for label, label_df in df.groupby('Label', sort=False):
                        remaining_rows = MAX_SAMPLES_PER_LABEL - label_counts.get(
                            label,
                            0,
                        )
                        if remaining_rows <= 0:
                            logger.info(
                                f"   - {label}: skipped {len(label_df)} rows; "
                                "cap already reached"
                            )
                            continue

                        selected_df = label_df.iloc[:remaining_rows].copy()
                        capped_parts.append(selected_df)
                        label_counts[label] = (
                            label_counts.get(label, 0) + len(selected_df)
                        )

                        dropped_rows = len(label_df) - len(selected_df)
                        if dropped_rows:
                            logger.info(
                                f"   - {label}: kept {len(selected_df)} rows "
                                f"from this file; dropped {dropped_rows} due to cap"
                            )

                    if capped_parts:
                        combined_dfs.append(pd.concat(capped_parts, ignore_index=True))
                else:
                    combined_dfs.append(df)
            except Exception as e:
                logger.error(f"❌ Error processing {filepath}: {e}")
                continue
        
        if not combined_dfs:
            raise ValueError("❌ No CSV files were successfully loaded!")
        
        # Combine all dataframes
        combined_df = pd.concat(combined_dfs, ignore_index=True)

        if NORMALIZE_LABELS:
            logger.info("\n🏷️ Normalizing equivalent labels...")
            for old_label, new_label in NORMALIZE_LABELS.items():
                count = (combined_df['Label'] == old_label).sum()
                if count:
                    logger.info(f"  - {old_label} -> {new_label}: {count} rows")
            combined_df['Label'] = combined_df['Label'].replace(NORMALIZE_LABELS)
        
        # Sort by timestamp, then cap each label with a contiguous time block.
        logger.info("\n📊 Sorting data by timestamp...")
        combined_df = combined_df.sort_values(by='Timestamp')

        if APPLY_MAX_SAMPLES_PER_LABEL and MAX_SAMPLES_PER_LABEL:
            logger.info(
                f"\n✂️ Applying per-label chronological block cap: "
                f"{MAX_SAMPLES_PER_LABEL} consecutive rows per label"
            )
            combined_df = cap_label_by_random_chronological_block(
                combined_df,
                label_column='Label',
                timestamp_column='Timestamp',
                max_samples_per_label=MAX_SAMPLES_PER_LABEL,
                random_state=RANDOM_STATE,
            )
        else:
            combined_df = (
                combined_df
                .sort_values(['Label', 'Timestamp'])
                .reset_index(drop=True)
            )
        
        # Log dataset statistics
        logger.info(f"\n✓ Total combined samples: {len(combined_df)}")
        logger.info(f"  Labels distribution:")
        for label, count in combined_df['Label'].value_counts().items():
            logger.info(f"    - {label}: {count} ({count/len(combined_df)*100:.2f}%)")
        
        logger.info(f"\n✓ Data loading completed successfully!")
        logger.info("="*60 + "\n")
        
        return combined_df


def load_data(kaggle_dataset_name: str, download_path: str, 
              csv_paths: Dict[str, str], auto_download: bool = True) -> pd.DataFrame:
    """
    Main function to download and load data
    
    Args:
        kaggle_dataset_name: Dataset name on Kaggle
        download_path: Where to download
        csv_paths: Mapping of protocols to relative file paths (will be joined with download_path)
        auto_download: If True, try to download from Kaggle
        
    Returns:
        Combined DataFrame with all data
    """
    download_path_obj = Path(download_path)
    from config.config import DATASET_NAME, DISCOVER_CSV_FILES

    if DATASET_NAME == "CICIoT2023" and DISCOVER_CSV_FILES:
        full_csv_paths = discover_ciciot2023_csv_files(download_path_obj)
        auto_download = False
    else:
        full_csv_paths = {}
        for attack_label, rel_path in csv_paths.items():
            rel_path = Path(rel_path)
            full_csv_paths[attack_label] = (
                str(rel_path) if rel_path.is_absolute()
                else str(download_path_obj / rel_path)
            )
    
    # Try to download from Kaggle if files don't exist
    if auto_download:
        kaggle_loader = KaggleDataLoader(kaggle_dataset_name, download_path)
        #if not any(os.path.exists(p) for p in full_csv_paths.values()):
        if not all(os.path.exists(p) for p in full_csv_paths.values()):
            logger.info("CSV files not found locally, attempting Kaggle download...")
            kaggle_loader.download_dataset()

    split_plan = None
    if DATASET_NAME == "CICIoT2023":
        from config.config import (
            MAX_SAMPLES_PER_LABEL,
            SOURCE_TIMESTAMP_COLUMN,
            TEST_SIZE,
            VALID_SIZE,
        )

        manifest = build_capture_manifest(
            full_csv_paths,
            timestamp_column=SOURCE_TIMESTAMP_COLUMN,
        )
        save_capture_manifest(
            manifest,
            BASE_DIR / "outputs" / "data" / "capture_manifest.csv",
        )
        split_plan = build_capture_split_plan(
            manifest,
            max_samples_per_label=MAX_SAMPLES_PER_LABEL,
            valid_size=VALID_SIZE,
            test_size=TEST_SIZE,
        )
        save_capture_split_plan(
            split_plan,
            BASE_DIR / "outputs" / "data" / "capture_split_plan.csv",
        )
    
    # Load data
    loader = DataLoader(full_csv_paths, split_plan=split_plan)
    data = loader.load_all_data()
    
    # output_path = Path("outputs/data")
    output_path = BASE_DIR / "outputs" / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    data.to_pickle(output_path / "step_1_Loaded_Data.pkl")
    logger.info(f"💾 Saved raw data to {output_path / 'step_1_Loaded_Data.pkl'}")

    return data

if __name__ == "__main__":
    import logging
    from pathlib import Path

    # ✅ Import config (same as main.py)
    from config.config import (
        KAGGLE_DATASET_NAME,
        DATA_DIR,
        CSV_PROTOCOLS,
        ALL_CSV_FILES,
        ACTIVE_CSV_FILES,
    )

    # -------------------- SETUP LOGGING --------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("\n")
    logger.info("█" * 80)
    logger.info("█" + " " * 20 + "STEP 1: LOAD DATA (STANDALONE)" + " " * 20 + "█")
    logger.info("█" * 80)

    # -------------------- SELECT WHICH FILES TO USE --------------------
    # Same behavior as main.py.
    csv_files = ACTIVE_CSV_FILES.copy()

    # -------------------- CHECK LOCAL DATA --------------------
    local_data_path = Path(DATA_DIR)

    if local_data_path.exists():
        logger.info(f"✓ Found local data at {local_data_path}")

        # Convert relative paths → full paths
        csv_files = {
            attack: str(local_data_path / rel_path)
            for attack, rel_path in csv_files.items()
        }
    else:
        logger.warning("⚠️ Local data not found, will try Kaggle download")

    # -------------------- LOAD DATA --------------------
    try:
        data = load_data(
            kaggle_dataset_name=KAGGLE_DATASET_NAME,
            download_path=str(DATA_DIR),
            csv_paths=csv_files,
            auto_download=True  # same as your main.py
        )

        # -------------------- OUTPUT (PIPELINE STYLE) --------------------
        logger.info("\n" + "=" * 60)
        logger.info("✓ DATA LOADED SUCCESSFULLY")
        logger.info("=" * 60)

        logger.info(f"Shape: {data.shape}")

        logger.info("\n📊 Label distribution:")
        for label, count in data['Label'].value_counts().items():
            logger.info(f" - {label}: {count} ({count/len(data)*100:.2f}%)")

        logger.info("\n🔍 First 5 rows:")
        logger.info(data.head())

        output_path = BASE_DIR / "outputs" / "data"
        output_path.mkdir(parents=True, exist_ok=True)
        save_file = output_path / "step_1_Loaded_Data.pkl"
        data.to_pickle(save_file)
        logger.info(f"\n💾 Saved loaded data to: {save_file}")

        logger.info("\n" + "=" * 60)
        logger.info("\n✅ STEP 1 COMPLETE")

    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise
