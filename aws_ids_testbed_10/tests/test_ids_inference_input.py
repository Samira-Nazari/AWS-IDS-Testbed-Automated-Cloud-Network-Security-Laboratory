import csv
import tempfile
import unittest
from pathlib import Path

from aws_ids_testbed_10.ids_inference_input import (
    prepare_inference_input,
)


def write_test_csv(path: Path, row_prefix: str, row_count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["record_id", "value"])

        for index in range(row_count):
            writer.writerow([f"{row_prefix}-{index + 1}", index + 1])


class TestInferenceInput(unittest.TestCase):
    def test_buffer_and_new_csv_are_combined_in_received_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer_path = root / "buffer.csv"
            new_csv_path = root / "new.csv"
            output_path = root / "benign_http_run01_job0002_inference.csv"

            write_test_csv(buffer_path, "buffer", 25)
            write_test_csv(new_csv_path, "new", 20)

            result = prepare_inference_input(
                buffer_csv_path=buffer_path,
                new_csv_path=new_csv_path,
                output_path=output_path,
            )

            self.assertEqual(result.total_rows, 45)
            self.assertEqual(result.plan.window_count, 4)
            self.assertEqual(result.plan.next_start_offset, 20)
            self.assertEqual(result.plan.retained_rows, 25)
            self.assertEqual(result.output_path, output_path)

            with output_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                rows = list(csv.reader(csv_file))[1:]

            self.assertEqual(len(rows), 45)
            self.assertEqual(rows[0][0], "buffer-1")
            self.assertEqual(rows[24][0], "buffer-25")
            self.assertEqual(rows[25][0], "new-1")
            self.assertEqual(rows[-1][0], "new-20")

    def test_fewer_than_30_rows_do_not_create_inference_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_csv_path = root / "new.csv"
            output_path = root / "inference.csv"

            write_test_csv(new_csv_path, "new", 20)

            result = prepare_inference_input(
                new_csv_path=new_csv_path,
                output_path=output_path,
            )

            self.assertFalse(result.plan.can_run_inference)
            self.assertIsNone(result.output_path)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
