import csv
import tempfile
import unittest
from pathlib import Path

from aws_ids_testbed_10.ids_stream_buffer_store import (
    commit_successful_buffer,
    save_waiting_buffer,
)
from aws_ids_testbed_10.ids_stream_continuity import (
    plan_continuous_windows,
)


def write_test_csv(path: Path, prefix: str, count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["record_id", "value"])

        for index in range(count):
            writer.writerow([f"{prefix}-{index + 1}", index + 1])


def read_record_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return [
            row["record_id"]
            for row in csv.DictReader(csv_file)
        ]


class TestStreamBufferStore(unittest.TestCase):
    def test_rows_below_30_are_saved_in_received_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer_path = root / "buffer.csv"
            csv1_path = root / "csv1.csv"
            csv2_path = root / "csv2.csv"

            write_test_csv(csv1_path, "csv1", 20)
            write_test_csv(csv2_path, "csv2", 9)

            first = save_waiting_buffer(
                new_csv_path=csv1_path,
                buffer_output_path=buffer_path,
            )

            second = save_waiting_buffer(
                existing_buffer_path=buffer_path,
                new_csv_path=csv2_path,
                buffer_output_path=buffer_path,
            )

            self.assertEqual(first.row_count, 20)
            self.assertEqual(second.row_count, 29)

            record_ids = read_record_ids(buffer_path)

            self.assertEqual(record_ids[0], "csv1-1")
            self.assertEqual(record_ids[19], "csv1-20")
            self.assertEqual(record_ids[20], "csv2-1")
            self.assertEqual(record_ids[-1], "csv2-9")

    def test_successful_inference_commits_only_retained_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inference_path = root / "inference.csv"
            buffer_path = root / "buffer.csv"

            write_test_csv(inference_path, "row", 45)
            plan = plan_continuous_windows(45)

            result = commit_successful_buffer(
                inference_input_path=inference_path,
                buffer_output_path=buffer_path,
                plan=plan,
            )

            self.assertEqual(plan.window_count, 4)
            self.assertEqual(plan.next_start_offset, 20)
            self.assertEqual(result.row_count, 25)

            record_ids = read_record_ids(buffer_path)

            self.assertEqual(record_ids[0], "row-21")
            self.assertEqual(record_ids[-1], "row-45")


if __name__ == "__main__":
    unittest.main()
