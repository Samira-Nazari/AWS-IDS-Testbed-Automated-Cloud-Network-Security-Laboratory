import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aws_ids_testbed_10.ids_inference_pipeline import (
    run_inference_steps_once,
)
from aws_ids_testbed_10.ids_inference_workspace import (
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)


class TestInferencePipeline(unittest.TestCase):
    def test_existing_steps_are_called_in_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = build_inference_workspace_paths(
                base_directory=Path(directory),
                scenario="benign_http",
            )
            create_inference_workspace_directories(paths)

            self.assertIsNotNone(paths.inference_csv_path)
            paths.inference_csv_path.write_text(
                "feature,value\n1,2\n",
                encoding="utf-8",
            )

            with (
                patch(
                    "aws_ids_testbed_10.ids_inference_pipeline."
                    "run_step_1"
                ) as step_1,
                patch(
                    "aws_ids_testbed_10.ids_inference_pipeline."
                    "run_step_2"
                ) as step_2,
                patch(
                    "aws_ids_testbed_10.ids_inference_pipeline."
                    "run_step_3"
                ) as step_3,
                patch(
                    "aws_ids_testbed_10.ids_inference_pipeline."
                    "run_step_4"
                ) as step_4,
            ):
                step_1.return_value = list(range(45))
                step_2.return_value = list(range(45))
                step_3.return_value = {
                    "X_windows": list(range(4)),
                }
                step_4.return_value = {
                    "summary": {"prediction_rows": 4},
                }

                result = run_inference_steps_once(
                    paths=paths,
                    batch_size=64,
                    device_setting="cpu",
                )

            step_1.assert_called_once()
            step_2.assert_called_once()
            step_3.assert_called_once()
            step_4.assert_called_once()

            self.assertEqual(result.input_rows, 45)
            self.assertEqual(result.processed_rows, 45)
            self.assertEqual(result.window_count, 4)
            self.assertEqual(result.prediction_count, 4)
            self.assertEqual(
                result.predictions_path,
                paths.current_predictions_path,
            )


if __name__ == "__main__":
    unittest.main()
