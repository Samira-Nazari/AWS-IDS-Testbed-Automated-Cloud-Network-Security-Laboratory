import json
import tempfile
import unittest
from pathlib import Path

from aws_ids_testbed_10.ids_inference_queue import (
    allocate_next_queue_sequence,
    build_inference_queue_job_paths,
    publish_inference_queue_job,
    reserve_inference_queue_job,
)


class TestInferenceQueue(unittest.TestCase):
    def test_atomically_publishes_prepared_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reservation = reserve_inference_queue_job(
                queue_directory=Path(directory),
                scenario="dos_http_flood",
                transaction_id=(
                    "dos_http_flood_20260915_120000"
                ),
            )
            reservation.staging_csv_path.write_text(
                "feature\n1\n",
                encoding="utf-8",
            )

            paths = publish_inference_queue_job(
                reservation=reservation,
                metadata={"window_count": 1},
            )

            self.assertFalse(
                reservation.staging_csv_path.exists()
            )
            self.assertTrue(paths.csv_path.is_file())
            self.assertTrue(paths.manifest_path.is_file())

            manifest = json.loads(
                paths.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["sequence"], 1)
            self.assertEqual(
                manifest["scenario"],
                "dos_http_flood",
            )
            self.assertEqual(manifest["window_count"], 1)
            self.assertEqual(
                Path(manifest["queue_csv_path"]),
                paths.csv_path,
            )

    def test_publish_requires_nonempty_staging_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reservation = reserve_inference_queue_job(
                queue_directory=Path(directory),
                scenario="benign_http",
                transaction_id=(
                    "benign_http_20260915_120000"
                ),
            )
            reservation.staging_csv_path.touch()

            with self.assertRaises(ValueError):
                publish_inference_queue_job(
                    reservation=reservation,
                    metadata={},
                )

    def test_reserves_hidden_staging_and_final_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reservation = reserve_inference_queue_job(
                queue_directory=Path(directory),
                scenario="dos_http_flood",
                transaction_id=(
                    "dos_http_flood_20260915_120000"
                ),
            )

            expected_csv_name = (
                "dos_http_flood_000001_"
                "20260915_120000_inference.csv"
            )

            self.assertEqual(reservation.sequence, 1)
            self.assertEqual(
                reservation.job_paths.csv_path.name,
                expected_csv_name,
            )
            self.assertEqual(
                reservation.staging_csv_path.name,
                f".{expected_csv_name}.preparing",
            )

    def test_reservation_rejects_existing_job_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue_directory = Path(directory)
            first = reserve_inference_queue_job(
                queue_directory=queue_directory,
                scenario="benign_http",
                transaction_id=(
                    "benign_http_20260915_120000"
                ),
            )

            sequence_path = queue_directory / ".last_sequence"
            sequence_path.write_text("0\n", encoding="utf-8")
            first.job_paths.csv_path.write_text(
                "data\n",
                encoding="utf-8",
            )

            with self.assertRaises(FileExistsError):
                reserve_inference_queue_job(
                    queue_directory=queue_directory,
                    scenario="benign_http",
                    transaction_id=(
                        "benign_http_20260915_120000"
                    ),
                )

    def test_allocates_persistent_sequence_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue_directory = Path(directory)

            self.assertEqual(
                allocate_next_queue_sequence(
                    queue_directory=queue_directory,
                ),
                1,
            )
            self.assertEqual(
                allocate_next_queue_sequence(
                    queue_directory=queue_directory,
                ),
                2,
            )
            self.assertEqual(
                (
                    queue_directory / ".last_sequence"
                ).read_text(encoding="utf-8"),
                "2\n",
            )

    def test_rejects_negative_stored_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue_directory = Path(directory)
            sequence_path = (
                queue_directory / ".last_sequence"
            )
            sequence_path.write_text(
                "-1\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                allocate_next_queue_sequence(
                    queue_directory=queue_directory,
                )

    def test_builds_scenario_first_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = build_inference_queue_job_paths(
                queue_directory=Path(directory),
                scenario="dos_http_flood",
                sequence=1,
                transaction_id=(
                    "dos_http_flood_20260915_120000"
                ),
            )

            expected_stem = (
                "dos_http_flood_000001_"
                "20260915_120000_inference"
            )

            self.assertEqual(
                paths.csv_path.name,
                f"{expected_stem}.csv",
            )
            self.assertEqual(
                paths.manifest_path.name,
                f"{expected_stem}.json",
            )

    def test_rejects_non_positive_sequence(self) -> None:
        with self.assertRaises(ValueError):
            build_inference_queue_job_paths(
                queue_directory=Path("queue"),
                scenario="benign_http",
                sequence=0,
                transaction_id=(
                    "benign_http_20260915_120000"
                ),
            )

    def test_rejects_unsafe_runtime_name(self) -> None:
        with self.assertRaises(ValueError):
            build_inference_queue_job_paths(
                queue_directory=Path("queue"),
                scenario="../attack",
                sequence=1,
                transaction_id="attack_20260915_120000",
            )


if __name__ == "__main__":
    unittest.main()
