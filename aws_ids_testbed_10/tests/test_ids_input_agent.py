import os
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aws_ids_testbed_10.ids_input_agent import (
    InputCandidate,
    find_next_completed_csv,
    prepare_selected_csv,
    run_input_agent_once,
)
from aws_ids_testbed_10.ids_inference_workspace import (
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)


class TestIdsInputAgent(unittest.TestCase):
    def _directories(self, directory: str):
        root = Path(directory)
        converted = root / "converted"
        csv_directory = root / "csv"
        state = root / "state"

        converted.mkdir()
        csv_directory.mkdir()

        for state_name in (
            "queued",
            "active",
            "completed",
            "failed",
        ):
            (state / state_name).mkdir(parents=True)

        return converted, csv_directory, state

    def _completed_capture(
        self,
        *,
        converted: Path,
        csv_directory: Path,
        transaction_id: str,
        modification_time: int,
    ) -> tuple[Path, Path]:
        marker = converted / f"{transaction_id}.pcap.done"
        csv_path = csv_directory / f"{transaction_id}.csv"

        marker.write_text("completed\n", encoding="utf-8")
        csv_path.write_text("flow,value\n1,2\n", encoding="utf-8")
        os.utime(marker, (modification_time, modification_time))

        return marker, csv_path

    def _write_flow_csv(
        self,
        path: Path,
        row_count: int,
        start: int = 0,
    ) -> None:
        pd.DataFrame(
            {
                "flow_id": range(start, start + row_count),
                "feature": range(
                    start + 100,
                    start + 100 + row_count,
                ),
            }
        ).to_csv(path, index=False)

    def test_selects_all_scenario_types_in_completion_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            converted, csv_directory, state = self._directories(
                directory
            )

            first_marker, first_csv = self._completed_capture(
                converted=converted,
                csv_directory=csv_directory,
                transaction_id=(
                    "dos_syn_flood_20260915_120000"
                ),
                modification_time=100,
            )
            self._completed_capture(
                converted=converted,
                csv_directory=csv_directory,
                transaction_id=(
                    "benign_http_20260915_120100"
                ),
                modification_time=200,
            )
            self._completed_capture(
                converted=converted,
                csv_directory=csv_directory,
                transaction_id=(
                    "dos_http_flood_20260915_120200"
                ),
                modification_time=300,
            )

            candidate = find_next_completed_csv(
                converted_marker_directory=converted,
                csv_directory=csv_directory,
                state_directory=state,
            )

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.scenario, "dos_syn_flood")
            self.assertEqual(
                candidate.transaction_id,
                "dos_syn_flood_20260915_120000",
            )
            self.assertEqual(
                candidate.converted_marker_path,
                first_marker,
            )
            self.assertEqual(candidate.csv_path, first_csv)

    def test_skips_missing_empty_and_already_handled_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            converted, csv_directory, state = self._directories(
                directory
            )

            missing_id = "benign_http_20260915_120000"
            missing_marker = converted / f"{missing_id}.pcap.done"
            missing_marker.write_text("completed\n", encoding="utf-8")
            os.utime(missing_marker, (100, 100))

            empty_id = "dos_syn_flood_20260915_120100"
            empty_marker = converted / f"{empty_id}.pcap.done"
            empty_marker.write_text("completed\n", encoding="utf-8")
            (csv_directory / f"{empty_id}.csv").touch()
            os.utime(empty_marker, (200, 200))

            handled_id = "dos_http_flood_20260915_120200"
            self._completed_capture(
                converted=converted,
                csv_directory=csv_directory,
                transaction_id=handled_id,
                modification_time=300,
            )
            (state / "completed" / f"{handled_id}.done").touch()

            expected_id = "future_attack_20260915_120300"
            _, expected_csv = self._completed_capture(
                converted=converted,
                csv_directory=csv_directory,
                transaction_id=expected_id,
                modification_time=400,
            )

            candidate = find_next_completed_csv(
                converted_marker_directory=converted,
                csv_directory=csv_directory,
                state_directory=state,
            )

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.scenario, "future_attack")
            self.assertEqual(candidate.transaction_id, expected_id)
            self.assertEqual(candidate.csv_path, expected_csv)

    def test_fewer_than_30_rows_are_committed_to_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            source_csv = root / "dos_http.csv"
            self._write_flow_csv(source_csv, 20)

            workspace = build_inference_workspace_paths(
                base_directory=root / "runtime",
                scenario="dos_http_flood",
            )
            create_inference_workspace_directories(workspace)

            candidate = InputCandidate(
                scenario="dos_http_flood",
                transaction_id=(
                    "dos_http_flood_20260915_120000"
                ),
                converted_marker_path=root / "capture.pcap.done",
                csv_path=source_csv,
            )

            result = prepare_selected_csv(
                candidate=candidate,
                workspace_paths=workspace,
                state_directory=state,
            )

            self.assertEqual(result.status, "waiting")
            self.assertEqual(result.combined_rows, 20)
            self.assertEqual(result.buffer_rows, 20)
            self.assertEqual(
                len(pd.read_csv(workspace.buffer_csv_path)),
                20,
            )
            self.assertFalse(
                (
                    state
                    / "active"
                    / f"{candidate.transaction_id}.active"
                ).exists()
            )
            self.assertTrue(
                (
                    state
                    / "completed"
                    / f"{candidate.transaction_id}.done"
                ).exists()
            )
            self.assertFalse(source_csv.exists())

    def test_queues_two_inputs_without_waiting_for_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            first_source = root / "dos_syn_first.csv"
            second_source = root / "dos_syn_second.csv"
            self._write_flow_csv(first_source, 10, start=25)
            self._write_flow_csv(second_source, 5, start=35)

            workspace = build_inference_workspace_paths(
                base_directory=root / "runtime",
                scenario="dos_syn_flood",
            )
            create_inference_workspace_directories(workspace)
            self._write_flow_csv(workspace.buffer_csv_path, 25)

            first_candidate = InputCandidate(
                scenario="dos_syn_flood",
                transaction_id=(
                    "dos_syn_flood_20260915_120100"
                ),
                converted_marker_path=root / "capture.pcap.done",
                csv_path=first_source,
            )

            first_result = prepare_selected_csv(
                candidate=first_candidate,
                workspace_paths=workspace,
                state_directory=state,
            )

            self.assertEqual(first_result.status, "queued")
            self.assertEqual(first_result.combined_rows, 35)
            self.assertEqual(first_result.buffer_rows, 25)
            self.assertEqual(
                len(pd.read_csv(workspace.buffer_csv_path)),
                25,
            )
            self.assertEqual(
                len(pd.read_csv(first_result.inference_csv_path)),
                35,
            )
            self.assertTrue(first_result.ready_job_path.exists())
            self.assertFalse(first_source.exists())

            job = json.loads(
                first_result.ready_job_path.read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(job["scenario"], "dos_syn_flood")
            self.assertEqual(job["window_count"], 2)
            self.assertEqual(job["retained_rows"], 25)

            second_candidate = InputCandidate(
                scenario="dos_syn_flood",
                transaction_id=(
                    "dos_syn_flood_20260915_120200"
                ),
                converted_marker_path=root / "second.pcap.done",
                csv_path=second_source,
            )

            second_result = prepare_selected_csv(
                candidate=second_candidate,
                workspace_paths=workspace,
                state_directory=state,
            )

            self.assertEqual(second_result.status, "queued")
            self.assertEqual(second_result.combined_rows, 30)
            self.assertEqual(second_result.buffer_rows, 25)
            self.assertEqual(
                len(pd.read_csv(workspace.buffer_csv_path)),
                25,
            )
            self.assertTrue(second_result.ready_job_path.exists())
            self.assertFalse(second_source.exists())
            self.assertTrue(first_result.ready_job_path.exists())

    def test_agent_cycle_prepares_a_completed_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            converted, csv_directory, state = self._directories(
                directory
            )
            transaction_id = "benign_http_20260915_120500"
            self._completed_capture(
                converted=converted,
                csv_directory=csv_directory,
                transaction_id=transaction_id,
                modification_time=100,
            )

            self._write_flow_csv(
                csv_directory / f"{transaction_id}.csv",
                20,
            )

            result = run_input_agent_once(
                base_directory=root / "runtime",
                converted_marker_directory=converted,
                csv_directory=csv_directory,
                state_directory=state,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.status, "waiting")
            self.assertEqual(result.transaction_id, transaction_id)
            self.assertEqual(result.buffer_rows, 20)

    def test_agent_cycle_does_not_wait_for_active_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            converted, csv_directory, state = self._directories(
                directory
            )
            (state / "active" / "older.active").touch()

            transaction_id = "benign_http_20260915_120600"
            self._completed_capture(
                converted=converted,
                csv_directory=csv_directory,
                transaction_id=transaction_id,
                modification_time=100,
            )
            self._write_flow_csv(
                csv_directory / f"{transaction_id}.csv",
                20,
            )

            result = run_input_agent_once(
                base_directory=root / "runtime",
                converted_marker_directory=converted,
                csv_directory=csv_directory,
                state_directory=state,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.status, "waiting")


if __name__ == "__main__":
    unittest.main()
