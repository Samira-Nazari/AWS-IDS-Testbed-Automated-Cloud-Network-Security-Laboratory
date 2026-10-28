from pathlib import Path

import pandas as pd

from config.config import ACTIVE_CSV_FILES, DATA_DIR


def main():
    all_samples = []

    for file_order, (config_name, rel_path) in enumerate(ACTIVE_CSV_FILES.items(), start=1):
        source_path = Path(DATA_DIR) / rel_path
        print(f"Reading: {source_path}")

        benign_parts = []
        non_benign_parts = []

        for chunk in pd.read_csv(source_path, chunksize=100_000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            chunk["Label"] = chunk["Label"].astype(str).str.strip()

            if sum(len(part) for part in benign_parts) < 5:
                benign_parts.append(chunk[chunk["Label"] == "BENIGN"].head(5).copy())

            if sum(len(part) for part in non_benign_parts) < 5:
                non_benign_parts.append(chunk[chunk["Label"] != "BENIGN"].head(5).copy())

            if (
                sum(len(part) for part in benign_parts) >= 5
                and sum(len(part) for part in non_benign_parts) >= 5
            ):
                break

        benign = pd.concat(benign_parts, ignore_index=True).head(5)
        non_benign = pd.concat(non_benign_parts, ignore_index=True).head(5)

        for sample_order, (sample_type, sample) in enumerate(
            [("BENIGN", benign), ("NON_BENIGN", non_benign)],
            start=1,
        ):
            sample.insert(0, "file_order", file_order)
            sample.insert(1, "sample_group_order", sample_order)
            sample.insert(2, "sample_type", sample_type)
            sample.insert(3, "config_name", config_name)
            sample.insert(4, "source_file", rel_path)
            sample.insert(5, "source_path", str(source_path))
            all_samples.append(sample)

    result = pd.concat(all_samples, ignore_index=True)

    out_path = Path("outputs/data/sample_5_benign_then_5_non_benign_per_file.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\nSaved to: {out_path}")
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")
    print("\nSummary:")
    print(result[["file_order", "source_file", "sample_type", "Label"]].value_counts().sort_index())


if __name__ == "__main__":
    main()
