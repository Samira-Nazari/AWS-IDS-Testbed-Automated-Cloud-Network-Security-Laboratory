import tempfile
import unittest
from pathlib import Path

from aws_ids_testbed_10.ids_inference_workspace import (
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)


class TestInferenceWorkspace(unittest.TestCase):
    def test_shared_workspace_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_directory = Path(directory) / "aws_ids_testbed"

            paths = build_inference_workspace_paths(
                base_directory=base_directory,
                scenario="benign_http_scenario4",
            )

            expected_directory = (
                base_directory / "inference"
            )

            self.assertEqual(
                paths.inference_directory,
                expected_directory,
            )
            self.assertEqual(
                paths.inference_csv_path,
                (
                    expected_directory
                    / "input"
                    / "benign_http_scenario4_inference.csv"
                ),
            )
            self.assertEqual(
                paths.queue_directory,
                expected_directory / "queue",
            )
            self.assertEqual(
                paths.buffer_csv_path,
                expected_directory / "stream_buffer.csv",
            )
            self.assertEqual(
                paths.current_scenario_execution_id_path,
                expected_directory
                / "current_scenario_execution_id.txt",
            )
            self.assertEqual(
                paths.current_predictions_path,
                expected_directory / "ids_predictions_current.csv",
            )
            self.assertEqual(
                paths.master_predictions_path,
                expected_directory / "ids_predictions.csv",
            )

            create_inference_workspace_directories(paths)

            self.assertTrue(paths.inference_directory.is_dir())
            self.assertTrue(paths.input_directory.is_dir())
            self.assertTrue(paths.queue_directory.is_dir())

            # Creating directories must not create output files.
            self.assertFalse(paths.buffer_csv_path.exists())
            self.assertFalse(paths.current_predictions_path.exists())
            self.assertFalse(paths.master_predictions_path.exists())

    def test_workspace_can_be_built_without_a_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = build_inference_workspace_paths(
                base_directory=Path(directory),
            )

            self.assertIsNone(paths.inference_csv_path)

    def test_unsafe_scenario_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_inference_workspace_paths(
                    base_directory=Path(directory),
                    scenario="../benign_http",
                )


if __name__ == "__main__":
    unittest.main()
