"""IDS inference STEP=1 data loader.

Based on Lite_V6_CICIoT2023/data/data_loader.py.

Original Lite V6 STEP=1:
- discover labeled CICIoT2023 CSV captures
- build capture manifest
- build train/validation/test split plan
- load planned rows
- save step_1_Loaded_Data.pkl

AWS IDS inference STEP=1:
- read converted IDS CSV files from artifacts/ids_csv
- infer expected scenario/label from filename for development checks
- build capture manifest
- load all rows for inference
- save step_1_Loaded_IDS_Data.pkl
"""

from __future__ import annotations

import csv
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ids_detector.lite_v6.config.config import (
    IDS_CAPTURE_MANIFEST_PATH,
    IDS_CSV_INPUT_DIR,
    IDS_TIMESTAMP_COLUMNS,
    LITE_V6_WINDOW_CONFIG_PATH,
    SCENARIO_TO_EXPECTED_LABEL,
    SOURCE_LABEL_COLUMN,
    SOURCE_TIMESTAMP_COLUMN,
    STEP_1_LOADED_DATA_PATH,
    TARGET_LABEL_COLUMN,
    TARGET_TIMESTAMP_COLUMN,
)


logger = logging.getLogger(__name__)


def discover_ids_csv_files(input_dir: Path = IDS_CSV_INPUT_DIR) -> Dict[str, str]:
    """Discover converted AWS IDS CSV files in one flat directory."""
    input_dir = Path(input_dir)
    csv_paths = {
        csv_file.stem: str(csv_file.resolve())
        for csv_file in sorted(input_dir.glob("*.csv"))
    }

    if not csv_paths:
        raise ValueError(f"No AWS IDS CSV files found under: {input_dir}")

    logger.info("Discovered %d AWS IDS CSV files under %s", len(csv_paths), input_dir)
    return csv_paths


def infer_expected_scenario(csv_path: Path) -> str:
    """Infer AWS testbed scenario from the converted CSV filename."""
    stem = csv_path.stem
    for scenario in sorted(SCENARIO_TO_EXPECTED_LABEL, key=len, reverse=True):
        if stem == scenario or stem.startswith(f"{scenario}_"):
            return scenario
    return "unknown"


def _parse_csv_record(raw_line: bytes) -> list:
    """Parse one CSV record read in binary mode."""
    text = raw_line.decode("utf-8-sig").rstrip("\r\n")
    return next(csv.reader([text]))


def _choose_timestamp_column(header: list[str], preferred_column: str) -> str:
    """Choose the IDS timestamp column to use as the main model timestamp."""
    candidates = [preferred_column, *IDS_TIMESTAMP_COLUMNS, TARGET_TIMESTAMP_COLUMN]
    for column in dict.fromkeys(candidates):
        if column in header:
            return column
    raise KeyError(
        f"No timestamp column found. Checked: {', '.join(dict.fromkeys(candidates))}"
    )


def _window_count_for_rows(row_count: int, window_size: int, step_size: int) -> int:
    """Calculate how many sliding windows can be created from consecutive rows."""
    if row_count < window_size:
        return 0
    return ((row_count - window_size) // step_size) + 1


def _rows_for_window_count(window_count: int, window_size: int, step_size: int) -> int:
    """Calculate the minimum rows needed to create a number of windows."""
    if window_count <= 0:
        return 0
    return window_size + ((window_count - 1) * step_size)


def _get_active_window_config() -> tuple[int, int]:
    """Load the window size and step size selected during Lite V6 training."""
    with Path(LITE_V6_WINDOW_CONFIG_PATH).open("rb") as input_file:
        window_config = pickle.load(input_file)

    window_size = int(window_config["best_window_size"])
    step_size = int(window_config["best_step_size"])

    if window_size <= 0 or step_size <= 0:
        raise ValueError(f"Invalid Lite V6 window config: {window_config}")

    return window_size, step_size


def inspect_capture_file(
    file_key: str,
    filepath: str,
    timestamp_column: str = SOURCE_TIMESTAMP_COLUMN,
) -> dict:
    """Collect exact row count and boundary timestamps for one IDS CSV file."""
    # Original Lite V6 behavior:
    # label = file_key.split("/", 1)[0]
    # AWS IDS uses flat CSV files, so the development label comes from filename.
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"IDS CSV not found: {path}")

    with path.open("rb") as csv_file:
        header_line = csv_file.readline()
        if not header_line:
            raise ValueError(f"IDS CSV is empty: {path}")

        header = _parse_csv_record(header_line)
        chosen_timestamp_column = _choose_timestamp_column(header, timestamp_column)
        timestamp_index = header.index(chosen_timestamp_column)

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
        raise ValueError(f"IDS CSV has no data rows: {path}")

    first_values = _parse_csv_record(first_record)
    last_values = _parse_csv_record(last_record)
    expected_columns = len(header)

    if len(first_values) != expected_columns or len(last_values) != expected_columns:
        raise ValueError(f"IDS CSV boundary row has bad column count: {path}")

    first_timestamp = float(first_values[timestamp_index])
    last_timestamp = float(last_values[timestamp_index])

    if not np.isfinite(first_timestamp) or not np.isfinite(last_timestamp):
        raise ValueError(f"IDS CSV has non-finite boundary timestamp: {path}")

    scenario = infer_expected_scenario(path)
    expected_label = SCENARIO_TO_EXPECTED_LABEL.get(scenario, "unknown")

    window_size, step_size = _get_active_window_config()
    model_window_count = _window_count_for_rows(row_count, window_size, step_size)

    return {
        "Label": expected_label,
        "expected_scenario": scenario,
        "expected_label_name": expected_label,
        "unique_id": f"IDS_{path.stem}",
        "file_path": str(path.resolve()),
        "file_name": path.name,
        "row_count": row_count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "timestamp_column": chosen_timestamp_column,
        "model_window_size": window_size,
        "model_step_size": step_size,
        "model_window_count": model_window_count,
        "can_make_model_window": model_window_count > 0,
    }


def build_capture_manifest(
    csv_paths: Dict[str, str],
    timestamp_column: str = SOURCE_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    """Build a chronological one-row-per-CSV IDS capture inventory."""
    # Original Lite V6 behavior also built this manifest, then used it for
    # train/validation/test planning. AWS IDS keeps only the manifest.
    logger.info("Building AWS IDS capture manifest...")
    records = []

    for capture_index, (file_key, filepath) in enumerate(csv_paths.items(), start=1):
        logger.info(
            "Inspecting IDS CSV %d/%d: %s",
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
        raise ValueError("IDS capture manifest is empty")

    if manifest["unique_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["unique_id"].duplicated(keep=False), "unique_id"
        ].tolist()
        raise ValueError(f"Duplicate IDS capture unique_id values found: {duplicates}")

    if (manifest["last_timestamp"] < manifest["first_timestamp"]).any():
        invalid_files = manifest.loc[
            manifest["last_timestamp"] < manifest["first_timestamp"],
            "file_path",
        ].tolist()
        raise ValueError(f"IDS timestamp range is reversed: {invalid_files}")

    manifest = manifest.sort_values(
        ["expected_scenario", "first_timestamp", "last_timestamp", "file_name"],
        kind="stable",
    ).reset_index(drop=True)

    manifest["chronological_order"] = (
        manifest.groupby("expected_scenario", sort=False).cumcount() + 1
    )

    manifest["previous_last_timestamp"] = manifest.groupby(
        "expected_scenario", sort=False
    )["last_timestamp"].shift(1)

    manifest["boundary_overlap"] = (
        manifest["previous_last_timestamp"].notna()
        & (manifest["first_timestamp"] <= manifest["previous_last_timestamp"])
    )

    logger.info(
        "IDS manifest complete: %d files, %d scenarios, %d rows, %d model windows",
        len(manifest),
        manifest["expected_scenario"].nunique(),
        int(manifest["row_count"].sum()),
        int(manifest["model_window_count"].sum()),
    )
    return manifest


def save_capture_manifest(
    manifest: pd.DataFrame,
    output_path: Path = IDS_CAPTURE_MANIFEST_PATH,
) -> None:
    """Save IDS capture inventory for later review and inference debugging."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    logger.info("Saved IDS capture manifest to: %s", output_path)


class DataLoader:
    """Load converted AWS IDS CSV files for Lite V6 inference."""

    def __init__(
        self,
        csv_paths: Dict[str, str],
        timestamp_column: str = SOURCE_TIMESTAMP_COLUMN,
    ) -> None:
        # Original Lite V6 __init__ stored CSV paths, timestamp format, and
        # optional split plan. AWS IDS inference stores only CSV paths and the
        # preferred timestamp column.
        self.csv_paths = csv_paths
        self.timestamp_column = timestamp_column

    @staticmethod
    def to_unix_timestamp(timestamp_value) -> Optional[float]:
        """Convert numeric or string timestamps into Unix timestamp seconds."""
        if pd.isna(timestamp_value):
            return None

        try:
            numeric_value = float(timestamp_value)
            if np.isfinite(numeric_value):
                return numeric_value
        except (TypeError, ValueError):
            pass

        parsed_timestamp = pd.to_datetime(timestamp_value, errors="coerce", utc=True)
        if pd.isna(parsed_timestamp):
            return None

        return float(parsed_timestamp.timestamp())

    def preprocess_csv(
        self,
        file_key: str,
        filepath: str,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read one IDS CSV and add the Lite V6 STEP=1 compatibility columns."""
        # Original Lite V6 preprocess_csv standardized Label/Timestamp columns,
        # removed invalid rows, added source metadata, and optionally loaded only
        # planned row ranges. AWS IDS reads the whole file for inference.
        path = Path(filepath)
        data = pd.read_csv(path, nrows=max_rows)

        if data.empty:
            logger.warning("Skipping empty IDS CSV: %s", path)
            return data

        chosen_timestamp_column = _choose_timestamp_column(
            list(data.columns),
            self.timestamp_column,
        )

        data[TARGET_TIMESTAMP_COLUMN] = data[chosen_timestamp_column].apply(
            self.to_unix_timestamp
        )
        data = data.dropna(subset=[TARGET_TIMESTAMP_COLUMN]).copy()

        scenario = infer_expected_scenario(path)
        expected_label = SCENARIO_TO_EXPECTED_LABEL.get(scenario, "unknown")

        data["expected_scenario"] = scenario
        data[SOURCE_LABEL_COLUMN] = expected_label
        data[TARGET_LABEL_COLUMN] = expected_label
        data["source_file_key"] = file_key
        data["source_file_name"] = path.name
        data["source_file_path"] = str(path.resolve())
        data["source_row_index"] = np.arange(len(data), dtype=np.int64)

        return data

    def load_all_data(self) -> pd.DataFrame:
        """Load and combine all converted IDS CSV files."""
        # Original Lite V6 load_all_data could follow a train/validation/test
        # split plan. AWS IDS inference keeps every available row.
        frames = []

        for file_key, filepath in self.csv_paths.items():
            logger.info("Loading IDS CSV: %s", filepath)
            frame = self.preprocess_csv(file_key=file_key, filepath=filepath)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            raise ValueError("No usable IDS CSV rows were loaded")

        data = pd.concat(frames, ignore_index=True)

        data = data.sort_values(
            ["expected_scenario", "source_file_name", TARGET_TIMESTAMP_COLUMN],
            kind="stable",
        ).reset_index(drop=True)

        logger.info("Loaded IDS inference data shape: %s", data.shape)
        logger.info(
            "Loaded IDS rows by expected label: %s",
            data[TARGET_LABEL_COLUMN].value_counts().to_dict(),
        )
        return data


def load_data(
    input_dir: Path = IDS_CSV_INPUT_DIR,
    output_path: Path = STEP_1_LOADED_DATA_PATH,
    manifest_path: Path = IDS_CAPTURE_MANIFEST_PATH,
) -> pd.DataFrame:
    """AWS IDS STEP=1 entry point."""
    # Original Lite V6 load_data discovered dataset files, built split plans,
    # loaded training data, and saved step_1_Loaded_Data.pkl. AWS IDS uses the
    # same entry-point idea but prepares inference data only.
    csv_paths = discover_ids_csv_files(Path(input_dir))

    manifest = build_capture_manifest(
        csv_paths=csv_paths,
        timestamp_column=SOURCE_TIMESTAMP_COLUMN,
    )
    save_capture_manifest(manifest, Path(manifest_path))

    loader = DataLoader(
        csv_paths=csv_paths,
        timestamp_column=SOURCE_TIMESTAMP_COLUMN,
    )
    data = loader.load_all_data()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output_file:
        pickle.dump(data, output_file)

    logger.info("Saved IDS STEP=1 loaded data to: %s", output_path)
    return data


def run_step_1() -> pd.DataFrame:
    """Run AWS IDS Lite V6 inference STEP=1 from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return load_data()


if __name__ == "__main__":
    run_step_1()
