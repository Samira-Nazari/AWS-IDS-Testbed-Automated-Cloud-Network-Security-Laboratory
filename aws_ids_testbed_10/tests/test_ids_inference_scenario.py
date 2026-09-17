import tempfile
import unittest
from pathlib import Path

from aws_ids_testbed_10.ids_inference_scenario import (
    read_scenario_execution_id,
    start_scenario_inference,
)
from aws_ids_testbed_10.ids_inference_workspace import (
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)


class TestInferenceScenario(unittest.TestCase):
    def test_start_scenario_resets_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_directory = Path(directory) / "aws_ids_testbed"

            old_paths = build_inference_workspace_paths(
                base_directory=base_directory,
                scenario="dos_http_flood",
            )

            create_inference_workspace_directories(old_paths)

            old_paths.buffer_csv_path.write_text(
                "old buffer",
                encoding="utf-8",
            )
            old_paths.step_1_path.write_text(
                "old step 1",
                encoding="utf-8",
            )
            old_paths.current_predictions_path.write_text(
                "old current predictions",
                encoding="utf-8",
            )
            old_paths.master_predictions_path.write_text(
                "cumulative predictions",
                encoding="utf-8",
            )

            self.assertIsNotNone(old_paths.inference_csv_path)
            old_paths.inference_csv_path.write_text(
                "old inference input",
                encoding="utf-8",
            )

            new_paths = start_scenario_inference(
                base_directory=base_directory,
                scenario="benign_http_scenario4",
                scenario_execution_id="execution-scenario-4",
            )

            self.assertEqual(
                new_paths.current_scenario_path.read_text(
                    encoding="utf-8"
                ),
                "benign_http_scenario4\n",
            )
            self.assertEqual(
                read_scenario_execution_id(new_paths),
                "execution-scenario-4",
            )

            self.assertFalse(new_paths.buffer_csv_path.exists())
            self.assertFalse(new_paths.step_1_path.exists())
            self.assertFalse(
                new_paths.current_predictions_path.exists()
            )
            self.assertFalse(
                old_paths.inference_csv_path.exists()
            )

            # The cumulative prediction CSV must be preserved.
            self.assertEqual(
                new_paths.master_predictions_path.read_text(
                    encoding="utf-8"
                ),
                "cumulative predictions",
            )

    def test_same_scenario_execution_also_resets_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_directory = Path(directory) / "aws_ids_testbed"

            paths = start_scenario_inference(
                base_directory=base_directory,
                scenario="benign_http",
            )
            first_execution_id = read_scenario_execution_id(paths)

            paths.buffer_csv_path.write_text(
                "old buffer",
                encoding="utf-8",
            )

            restarted_paths = start_scenario_inference(
                base_directory=base_directory,
                scenario="benign_http",
            )
            second_execution_id = read_scenario_execution_id(
                restarted_paths
            )

            self.assertFalse(
                restarted_paths.buffer_csv_path.exists()
            )
            self.assertTrue(
                restarted_paths.master_predictions_path.parent.is_dir()
            )
            self.assertNotEqual(
                first_execution_id,
                second_execution_id,
            )


if __name__ == "__main__":
    unittest.main()
