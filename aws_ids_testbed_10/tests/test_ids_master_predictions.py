import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aws_ids_testbed_10.ids_master_predictions import (
    update_master_predictions,
)


class TestMasterPredictions(unittest.TestCase):
    def test_predictions_are_accumulated_without_retry_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            current_path = directory_path / "ids_predictions_current.csv"
            master_path = directory_path / "ids_predictions.csv"

            pd.DataFrame(
                {
                    "window_index": [0, 1],
                    "predicted_label_name": [
                        "Benign_Final",
                        "DoS-HTTP_Flood",
                    ],
                    "confidence": [0.98, 0.97],
                }
            ).to_csv(current_path, index=False)

            first_result = update_master_predictions(
                current_predictions_path=current_path,
                master_predictions_path=master_path,
                scenario="benign_http",
                scenario_execution_id="execution-1",
                transaction_id="transaction-1",
                inference_csv_name="benign_http_inference.csv",
                diagnosed_at_utc="2026-09-15T12:00:00Z",
            )

            self.assertEqual(first_result.added_rows, 2)
            self.assertEqual(first_result.total_rows, 2)

            retry_result = update_master_predictions(
                current_predictions_path=current_path,
                master_predictions_path=master_path,
                scenario="benign_http",
                scenario_execution_id="execution-1",
                transaction_id="transaction-1",
                inference_csv_name="benign_http_inference.csv",
                diagnosed_at_utc="2026-09-15T12:01:00Z",
            )

            self.assertEqual(retry_result.added_rows, 0)
            self.assertEqual(retry_result.total_rows, 2)

            second_result = update_master_predictions(
                current_predictions_path=current_path,
                master_predictions_path=master_path,
                scenario="benign_http",
                scenario_execution_id="execution-1",
                transaction_id="transaction-2",
                inference_csv_name="benign_http_inference.csv",
                diagnosed_at_utc="2026-09-15T12:02:00Z",
            )

            self.assertEqual(second_result.added_rows, 2)
            self.assertEqual(second_result.total_rows, 4)

            master = pd.read_csv(master_path)
            self.assertEqual(len(master), 4)
            self.assertEqual(master["prediction_id"].nunique(), 4)
            self.assertEqual(
                master.columns[:6].tolist(),
                [
                    "prediction_id",
                    "scenario",
                    "scenario_execution_id",
                    "transaction_id",
                    "inference_csv_name",
                    "diagnosed_at_utc",
                ],
            )


if __name__ == "__main__":
    unittest.main()
