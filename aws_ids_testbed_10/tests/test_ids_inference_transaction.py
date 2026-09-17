import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from aws_ids_testbed_10.ids_inference_pipeline import (
    InferencePipelineResult,
)
from aws_ids_testbed_10.ids_inference_transaction import (
    process_received_csv,
)
from aws_ids_testbed_10.ids_inference_workspace import (
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)
from aws_ids_testbed_10.ids_master_predictions import (
    MasterPredictionResult,
)


def _write_csv(path: Path, row_count: int, start: int = 0) -> None:
    pd.DataFrame(
        {
            "flow_id": range(start, start + row_count),
            "feature": range(start + 100, start + 100 + row_count),
        }
    ).to_csv(path, index=False)


class TestInferenceTransaction(unittest.TestCase):
    def _paths(self, directory: str):
        paths = build_inference_workspace_paths(
            base_directory=Path(directory),
            scenario="benign_http",
        )
        return create_inference_workspace_directories(paths)

    def _pipeline_result(self, paths, prediction_count: int = 2):
        return InferencePipelineResult(
            input_rows=35,
            processed_rows=35,
            window_count=2,
            prediction_count=prediction_count,
            predictions_path=paths.current_predictions_path,
        )

    def _master_result(self, paths, total_rows: int = 2):
        return MasterPredictionResult(
            added_rows=2,
            total_rows=total_rows,
            master_predictions_path=paths.master_predictions_path,
        )

    def test_waits_when_fewer_than_30_rows_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            new_csv_path = Path(directory) / "received.csv"
            _write_csv(new_csv_path, 20)

            with patch(
                "aws_ids_testbed_10.ids_inference_transaction."
                "run_inference_steps_once"
            ) as pipeline:
                result = process_received_csv(
                    new_csv_path=new_csv_path,
                    paths=paths,
                    scenario="benign_http",
                    scenario_execution_id="execution-1",
                    transaction_id="transaction-1",
                )

            pipeline.assert_not_called()
            self.assertEqual(result.status, "waiting")
            self.assertEqual(result.attempts, 0)
            self.assertEqual(result.buffer_rows, 20)
            self.assertEqual(len(pd.read_csv(paths.buffer_csv_path)), 20)

    def test_success_commits_predictions_and_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            new_csv_path = Path(directory) / "received.csv"
            _write_csv(new_csv_path, 35)

            with (
                patch(
                    "aws_ids_testbed_10.ids_inference_transaction."
                    "run_inference_steps_once",
                    return_value=self._pipeline_result(paths),
                ) as pipeline,
                patch(
                    "aws_ids_testbed_10.ids_inference_transaction."
                    "update_master_predictions",
                    return_value=self._master_result(paths),
                ) as update_master,
            ):
                result = process_received_csv(
                    new_csv_path=new_csv_path,
                    paths=paths,
                    scenario="benign_http",
                    scenario_execution_id="execution-1",
                    transaction_id="transaction-1",
                )

            pipeline.assert_called_once()
            update_master.assert_called_once()
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.attempts, 1)
            self.assertEqual(result.prediction_rows, 2)
            self.assertEqual(result.buffer_rows, 25)
            self.assertEqual(len(pd.read_csv(paths.buffer_csv_path)), 25)

    def test_pipeline_is_retried_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            new_csv_path = Path(directory) / "received.csv"
            _write_csv(new_csv_path, 35)

            with (
                patch(
                    "aws_ids_testbed_10.ids_inference_transaction."
                    "run_inference_steps_once",
                    side_effect=[
                        RuntimeError("first attempt failed"),
                        self._pipeline_result(paths),
                    ],
                ) as pipeline,
                patch(
                    "aws_ids_testbed_10.ids_inference_transaction."
                    "update_master_predictions",
                    return_value=self._master_result(paths),
                ),
            ):
                result = process_received_csv(
                    new_csv_path=new_csv_path,
                    paths=paths,
                    scenario="benign_http",
                    scenario_execution_id="execution-1",
                    transaction_id="transaction-1",
                )

            self.assertEqual(pipeline.call_count, 2)
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.attempts, 2)
            self.assertIsNone(result.error)

    def test_two_failures_preserve_the_previous_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            _write_csv(paths.buffer_csv_path, 10)

            new_csv_path = Path(directory) / "received.csv"
            _write_csv(new_csv_path, 25, start=10)

            with (
                patch(
                    "aws_ids_testbed_10.ids_inference_transaction."
                    "run_inference_steps_once",
                    side_effect=RuntimeError("model failed"),
                ) as pipeline,
                patch(
                    "aws_ids_testbed_10.ids_inference_transaction."
                    "update_master_predictions"
                ) as update_master,
            ):
                result = process_received_csv(
                    new_csv_path=new_csv_path,
                    paths=paths,
                    scenario="benign_http",
                    scenario_execution_id="execution-1",
                    transaction_id="transaction-1",
                )

            self.assertEqual(pipeline.call_count, 2)
            update_master.assert_not_called()
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.attempts, 2)
            self.assertEqual(result.buffer_rows, 10)
            self.assertEqual(len(pd.read_csv(paths.buffer_csv_path)), 10)
            self.assertIn("RuntimeError: model failed", result.error)


if __name__ == "__main__":
    unittest.main()
