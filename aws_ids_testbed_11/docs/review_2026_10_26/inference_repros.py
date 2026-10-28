"""Local reproductions from the 2026-10-26 inference review.

Run from the project root with: python -B docs/review_2026_10_26/inference_repros.py
Uses temporary runtime directories; never accesses AWS.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

from aws_ids_testbed_11.ids_input_agent import InputCandidate, prepare_selected_csv
from aws_ids_testbed_11.ids_inference_agent import run_inference_agent_once
from aws_ids_testbed_11.ids_inference_pipeline import InferencePipelineResult
from aws_ids_testbed_11.ids_inference_scenario import start_inference_run
from aws_ids_testbed_11.ids_inference_workspace import (
    build_inference_workspace_paths,
    create_inference_workspace_directories,
)
from aws_ids_testbed_11.ids_scenario_lifecycle import (
    inspect_current_scenario_drain_state,
    mark_scenario_input_finished,
    scenario_quiet_period_elapsed,
)


def write_rows(path, start, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["flow_id", "feature"])
        writer.writerows((i, i) for i in range(start, start + count))


def read_ids(path):
    with path.open() as source:
        return [int(row["flow_id"]) for row in csv.DictReader(source)]


def publication_failure():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = create_inference_workspace_directories(
            build_inference_workspace_paths(
                base_directory=root, scenario="dos_syn_flood"
            )
        )
        state = root / "state" / "ids_input_agent"
        source = root / "source.csv"
        write_rows(workspace.buffer_csv_path, 0, 25)
        write_rows(source, 25, 10)
        candidate = InputCandidate(
            "dos_syn_flood", "dos_syn_flood_20261026_120000",
            root / "capture.pcap.done", source,
        )
        with patch(
            "aws_ids_testbed_11.ids_input_agent.publish_inference_queue_job",
            side_effect=OSError("injected publish failure"),
        ):
            try:
                prepare_selected_csv(
                    candidate=candidate, workspace_paths=workspace,
                    state_directory=state,
                )
            except OSError:
                pass
        print("PUBLISH FAILURE: source retained =", source.exists(),
              "; buffer =", read_ids(workspace.buffer_csv_path))
        result = prepare_selected_csv(
            candidate=candidate, workspace_paths=workspace,
            state_directory=state,
        )
        print("RETRIED JOB: rows =", read_ids(result.inference_csv_path),
              "; expected =", list(range(35)))


def interrupted_consumer():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = create_inference_workspace_directories(
            build_inference_workspace_paths(
                base_directory=root, scenario="dos_syn_flood"
            )
        )
        workspace.current_scenario_execution_id_path.write_text("test_execution")
        state = root / "state" / "ids_input_agent"
        for minute, start in [(0, 0), (1, 30)]:
            source = root / f"source_{minute}.csv"
            write_rows(source, start, 30)
            prepare_selected_csv(
                candidate=InputCandidate(
                    "dos_syn_flood", f"dos_syn_flood_20261026_120{minute}00",
                    root / "capture.pcap.done", source,
                ),
                workspace_paths=workspace, state_directory=state,
            )
        with patch(
            "aws_ids_testbed_11.ids_inference_agent.run_inference_steps_once",
            side_effect=KeyboardInterrupt("simulated process termination"),
        ):
            try:
                run_inference_agent_once(base_directory=root, state_directory=state)
            except KeyboardInterrupt:
                pass
        print("AFTER TERMINATION: ready jobs =",
              len(list(workspace.queue_directory.glob("*.json"))),
              "; active markers =",
              [p.name for p in (state / "active").glob("*.active")])
        try:
            run_inference_agent_once(base_directory=root, state_directory=state)
        except RuntimeError as error:
            print("RESTART RESULT:", str(error))


def scenario_isolation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = start_inference_run(base_directory=root, execution_id="old_run")
        state = root / "state" / "ids_input_agent"
        source = root / "old.csv"
        write_rows(source, 0, 30)
        prepared = prepare_selected_csv(
            candidate=InputCandidate(
                "benign_http", "benign_http_20261026_120000",
                root / "old.pcap.done", source,
            ),
            workspace_paths=workspace, state_directory=state,
        )
        start_inference_run(base_directory=root, execution_id="new_run")
        mark_scenario_input_finished(base_directory=root, execution_id="new_run")
        drain = inspect_current_scenario_drain_state(base_directory=root)
        print("RESET/QUEUE: previous run queue preserved =",
              prepared.ready_job_path.exists(),
              "; old manifest has execution id =",
              json.loads(prepared.ready_job_path.read_text()).get("scenario_execution_id"))
        print("DRAIN: pipeline_busy =", drain.pipeline_busy,
              "; ready_for_quiet_period =", drain.ready_for_quiet_period,
              "; quiet_seconds=0 =>",
              scenario_quiet_period_elapsed(drain_state=drain, quiet_seconds=0))

        def fake_pipeline(*, paths):
            paths.current_predictions_path.write_text(
                "window_index,predicted_label_name,predicted_label_id,confidence\n"
                "0,Benign_Final,0,1.0\n"
            )
            return InferencePipelineResult(30, 30, 1, 1, paths.current_predictions_path)

        with patch(
            "aws_ids_testbed_11.ids_inference_agent.run_inference_steps_once",
            side_effect=fake_pipeline,
        ):
            run_inference_agent_once(base_directory=root, state_directory=state)
        with workspace.master_predictions_path.open() as source:
            row = next(csv.DictReader(source))
        print("CONSUMER: previous run transaction assigned to execution =",
              row["scenario_execution_id"])


publication_failure()
interrupted_consumer()
scenario_isolation()
