# AWS IDS Testbed 11 — project review

Reviewed on October 26, 2026. Scope: the project at
`/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11`.

The project passes its existing unit tests and can execute the four inference
steps with the bundled model on CPU. However, reproduced failures can lose
captured traffic, corrupt stream continuity, block inference after a restart,
and assign previous-run traffic to a new experiment. These should be corrected
before relying on new experiment results.

This is a local source and artifact review, not certification of the running
AWS deployment. No AWS requests, SSH connections, deployments, packet capture,
traffic generation, training, downloads, or dependency installation were
performed. Application source and existing experiment artifacts were not
edited. This report and two reproduction scripts were added under `docs/`.

## Verification results

| Check | Result and limits |
| --- | --- |
| Python syntax | All 94 original Python files in the application, tests, detector, converter, and training project parsed successfully. |
| Shell syntax | All 17 shell/Slurm launch scripts passed `bash -n`. This does not establish remote runtime behavior. |
| Existing unit tests | `python3 -B -m unittest discover -s tests -v`: 39 tests passed with the system interpreter. Deployment tests use mocks. |
| Bundled virtual environment | Unusable for the documented workflow: metadata specifies Python 3.10.9, but its interpreter symlink now runs Python 3.13.9. It has no importable pip, YAML, NumPy, pandas, or other required packages. Test discovery there produced 20 passing tests and six module import errors. |
| System environment | CLI `--help` and `status` succeed. FastAPI, Uvicorn, python-multipart, dpkt, and Scapy are absent. `pip check` reports botocore/aiobotocore and docutils/Sphinx conflicts; these concern the shared system environment, not a validated project installation. |
| Real inference smoke | Actual Steps 1–4 on temporary synthetic data: 35 input/processed rows, 2 windows, 2 predictions, 34 probability columns, one CPU model batch. Model inference was not mocked. This establishes execution compatibility, not detection accuracy. |
| Model artifact consistency | Checkpoint loads; finite CPU logits have shape `(1, 34)` for `(1, 30, 74)` input. Scaler feature order matches all 74 selected features. Window size is 30, stride 5. Label encoder IDs `0..33` match the 34 saved class names. |
| Static lint | `ruff check --select F,E9` reports 180 diagnostics, predominantly unused imports/variables, redundant f-strings, and star-import ambiguity. No syntax failure was found. No automatic fixes applied. |
| AWS inventory | Saved inventory lists all eight roles as terminated, without current addresses. This was not checked against AWS. |

The existing tests cover useful queue, buffer, workspace, and deployment
behavior. They do not cover the consumer's interruption recovery or the three
failure cases demonstrated below. There is no dedicated
`test_ids_inference_agent.py`.

## Findings requiring attention before further experiments

Priority P1 means correct before trusting further experiment output. P2 means
address in the next reliability/reproducibility pass. Priorities describe
impact and prerequisites; they are not claims that every failure occurred in
the saved experiments.

### 1. P1 — Stopping capture can strand the final PCAP chunk

Primary location: [victim_capture_agent_service.py:149](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/victim_capture_agent_service.py:149).

The stop command sends a signal only to the agent shell and immediately
removes its PID file. The shell runs `sudo timeout ... tcpdump` in the
foreground, has no termination handler, and publishes the PCAP only after the
capture command returns (`scripts/continuous_capture_agent.sh:71`, `:94`).
Terminating the shell can leave its capture child running and its final chunk
in `/tmp`, outside the sender's pending directory.

The normal scenario path then drains uploads and declares input finished
(`cli.py:859`). Delivery verification checks pending/sent files and accepts
historical sent files matching the scenario; it cannot see this stranded chunk
(`victim_sender_agent_service.py:213`). A scenario can finish with missing
traffic.

Evidence: the actual capture script was run with temporary paths and harmless
mock capture commands. The shell exited with `-15` while its child remained
alive. No real packets were captured.

Suggested correction: handle termination, stop and wait for the capture child,
publish the completed final chunk and metadata, then acknowledge shutdown.
Verify delivery against the current execution's expected captures.

### 2. P1 — Failed queue publication corrupts flow continuity

Primary location: [ids_input_agent.py:323](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/ids_input_agent.py:323).

Agent 1 advances `stream_buffer.csv` before publishing the durable queue job.
If publication raises an exception, the handler at line 364 removes a queued
marker but does not restore the buffer. A retry combines the source with an
already advanced buffer.

Evidence: with buffered rows `0..24` and new rows `25..34`, an injected
publication `OSError` left the buffer as `10..34`. Retrying queued
`[10..34, 25..34]`, instead of `0..34`: rows disappeared and others repeated.
A process interruption after the queued marker is written but before job
publication can instead cause the source to be skipped permanently.

Suggested correction: persist an explicit preparation transaction and recover
publication and buffer advancement together. Merely reversing the writes
would introduce a different interruption window.

### 3. P1 — An interrupted inference job blocks subsequent jobs after restart

Primary location: [ids_inference_agent.py:320](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/ids_inference_agent.py:320).

The consumer moves a job out of the ready queue and creates an active marker.
There is no startup recovery for the claimed manifest, CSV, or marker. The
guard at line 290 rejects all later work while any active marker exists.

Evidence: interrupting the first of two queued jobs after claiming it left one
ready job and an active marker. Restarting processing raised
`Another inference transaction is active`. A normal restart neither resumes
the claimed job nor processes the remaining one.

Suggested correction: recover claimed jobs on startup, distinguish live
ownership from stale state, and use prediction IDs to make replay safe.

### 4. P1 — Previous-run jobs can be attributed to the next experiment

Primary location: [ids_inference_agent.py:339](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/ids_inference_agent.py:339).

`start_inference_run` resets shared inputs and artifacts but preserves queued
jobs (`ids_inference_scenario.py:122`). Their manifests contain no originating
execution ID (`ids_input_agent.py:335`). The consumer assigns the current
execution ID when it eventually processes them.

The drain checker also scans obsolete `state/ids_input_agent/ready`, whereas
the real queue is `inference/queue` (`ids_scenario_lifecycle.py:235`).

Evidence: a job queued in `old_run` survived reset and was recorded in the
master predictions as `new_run`. While that job remained queued, drain
inspection reported `pipeline_busy=False` and
`ready_for_quiet_period=True`.

Suggested correction: attach an immutable execution ID at receipt/queueing,
define how pending work is handled across runs, and inspect the actual queue
when deciding completion.

### 5. P1 — Converter direction statistics are incorrect

Primary location: [Supporting_functions.py:124](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/ids_pcap_to_csv/pcap2csv/Supporting_functions.py:124).

The extractor sorts endpoints before storing TCP and UDP packets
(`Feature_extraction.py:293`, `:338`), combining both directions under one
key. `get_src_dst_packets` searches for a reversed key that the extractor
never creates. Forward counters therefore contain both directions, while
reverse counters and `drate` remain zero for distinct endpoints.

Evidence: one A-to-B packet of 60 bytes and one B-to-A packet of 80 bytes
returned `(2, 0, 140, 0)` from the actual helper. Directional counts/bytes
should distinguish the two packets. These fields are selected by the model.

Suggested correction: retain packet direction relative to a defined flow
origin and verify counters with a small bidirectional capture. Validate the
corrected feature contract against training before reusing the checkpoint.

### 6. P1 — Scenario validation can report success without successful diagnosis

Primary location: [ids_inference_agent_service.py:177](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/ids_inference_agent_service.py:177).

The completion waiter accepts a `.complete.json` file without examining its
failure counts. Agent 1 can deliberately write such a marker after failed
transactions drain (`ids_input_agent.py:474`). The subsequent prediction check
only requires the cumulative CSV to be nonempty; it prints a row count but
does not require positive rows or rows for the current execution
(`ids_inference_agent_service.py:213`). Old results can satisfy this check.
Completion inspection also does not verify alert processing, despite the
stronger completion claim in the report speech.

Evidence: the generated verification command returned status `0` for a
header-only prediction CSV and printed `Prediction rows: 0`. The generated
wait command also returned `0` for a completion marker containing a conversion
failure. Commands ran against temporary directories with SSH mocked out.

Suggested correction: distinguish drained, successful, partially failed, and
failed runs; check current-run predictions and expected alert IDs. Preserve
failure details in the CLI exit status and reports.

## Additional confirmed reliability findings

### 7. P2 — Several duration fields contain absolute dates

Primary location: [Supporting_functions.py:42](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/ids_pcap_to_csv/pcap2csv/Supporting_functions.py:42).

Min/max/sum/mean "duration" are calculated from absolute packet timestamps;
singleton `idle_time` is also the timestamp itself. For timestamps
`1700000000` and `1700000001`, flow duration is 1, but minimum duration is
1700000000, maximum is 1700000001, sum is 3400000001, and mean is
1700000000.5. A single packet has idle time 1700000000. Identical traffic
shifted to another date therefore changes these features.

The saved scaler contains the same epoch-scale behavior, so simply changing
extraction would also change the trained contract. The enabled diagnostic
neutralization replaces these five fields with scaler means before inference
(`ids_detector/lite_v6/config/config.py:90`,
`features/feature_engineering.py:234`). This is an intentional mitigation,
documented in `ReportSpeech.md:293`, rather than another implementation bug.
It nevertheless makes live preprocessing differ from training.

Suggested correction: define timing features explicitly, record the diagnostic
setting with experiment results, and validate or retrain with consistent
training/inference extraction.

### 8. P2 — Repeating a core-instance create command loses the previous instance ID

Primary location: [cli.py:238](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/cli.py:238).

Victim, attacker, and IDS creation always launch a new instance and replace
that role's inventory entry. Repeating creation while an old instance runs
leaves it outside normal inventory-based termination. The benign-generator
path already checks for existing roles.

Evidence: two calls to the actual victim creation function with mocked EC2 and
temporary inventory produced two launches, but retained only `mocked-2`.
No billable resource was created during review.

Suggested correction: guard existing live roles, reconcile state, and preserve
all created IDs until termination is confirmed.

### 9. P2 — Alert recovery cannot deduplicate from its own event log

Primary location: [ids_alert_agent.py:255](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/ids_alert_agent.py:255).

Alert events omit `prediction_id`, execution ID, and transaction ID because
those fields are commented out. `load_seen_ids` expects `prediction_id` when
recovering from events. A crash after appending events and before saving
processed IDs can produce duplicates on restart. An existing but stale state
file also bypasses event-log reconciliation. The separate
`ids_alert_events.py:79` event writer similarly omits correlation IDs.

Evidence: injecting an error in `save_seen_ids` after one event was written
left one event, recovered zero IDs from that log, and appended the same
prediction again on restart.

Suggested correction: retain correlation IDs in stored events and reconcile
event and state progress when recovering. Keep display formatting separate
from the durable event schema.

### 10. P2 — PCAP visibility precedes timing-metadata visibility

Primary location: [continuous_capture_agent.sh:94](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/scripts/continuous_capture_agent.sh:94).

Capture publishes the PCAP before creating its timing sidecar at line 113.
The concurrent sender can upload it first and silently substitute `{}` for
missing metadata (`send_pcap_to_ids.sh:42`). This loses capture timestamps
and can leave orphan sidecars. The receiver repeats the ordering by exposing
the PCAP at `ids_receiver_app.py:73`, then metadata at line 122.

Evidence: a local receiver-function reproduction with framework imports
stubbed observed the published PCAP while its sidecar was absent. The victim
sender interleaving follows from source order and was not stress-tested.

Suggested correction: publish metadata before the file that triggers consumers,
or use a final readiness manifest representing the complete pair.

### 11. P2 — Converter verification masks failed file checks

Primary location: [ids_pcap_converter_service.py:118](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/ids_pcap_converter_service.py:118).

The `&&` chain checking the agent, permissions, timing helper, and shell syntax
ends with a semicolon. Successful subsequent Python checks mask earlier
failure. A local execution with both the agent and timing helper absent and
a mocked healthy Python runtime returned `0`.

Suggested correction: explicitly propagate each failed prerequisite rather
than relying on the final command's exit code.

### 12. P2, conditional — Filename reuse can invalidate capture/result consistency

Primary location: [ids_receiver_app.py:62](/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11/aws_ids_testbed_11/ids_receiver_app.py:62).

Uploads replace an existing filename without checking identity/content, but
converter done/failed markers are keyed only by filename and suppress later
conversion (`pcap_to_csv_agent.sh:103`). Different bytes uploaded under a
processed name can leave the PCAP inconsistent with its CSV and predictions;
replacing a failed input also does not retry it. This requires name reuse,
such as a manual resend or collision.

Evidence: two local receiver-function uploads named `same.pcap` replaced the
original content. HTTP framework imports were stubbed, so this validates the
function's storage behavior, not the complete HTTP stack.

Suggested correction: define idempotent receipt using a capture ID and content
hash; reject conflicting duplicates or create a new version with coordinated
conversion state.

## Saved experiment artifacts

The three bundled converted CSVs contain 216 benign HTTP rows, 4,006 HTTP
flood rows, and 303 SYN flood rows, each with 77 columns. The saved window
array is finite float32 data with shape `(889, 30, 74)`.

The cumulative prediction CSV contains 1,027 unique prediction IDs, 34 finite
probability columns, and probability sums within approximately `2.2e-7` of 1.
Its stored scenario-label agreement is:

| Stored scenario | Windows | Predictions matching declared class | Agreement |
| --- | ---: | ---: | ---: |
| HTTP flood | 969 | 931 | 96.08% |
| SYN flood | 58 | 57 | 98.28% |

These are calculations from saved cumulative predictions and scenario-derived
expected labels, not fresh or independently labeled evaluation results. The
CSV has no benign-scenario predictions, so it cannot establish benign false
positive performance. The cumulative row count need not match the most recent
replaceable window array. Training/test arrays and evaluation results are
absent from the bundled training outputs, preventing independent reproduction
of trained-model quality during this review.

## Scope and remaining limits

Reviewed configuration, inventory, EC2 lifecycle, SSH helpers, deployment,
attacker and benign traffic services, victim capture/sending, receiver,
converter, input/consumer queues, buffering, transactions, cumulative
predictions, alerts, timing reports, metrics, CLI orchestration, tests, saved
model artifacts, and documentation. Training review included split planning,
capture/row boundaries, scaler fitting, feature selection, train-only SMOTE,
label encoding, sampler/loss settings, checkpoint selection, and evaluation.
No additional confirmed active-path training defect was identified.

Live IAM permissions, security-group rules, subnet routing, AMI availability
and compatibility, installed EC2 dependencies, remote service health,
capture loss under load, and complete PCAP-to-CSV execution remain unverified.
The local packet-processing dependencies are missing. The documented lab
receiver binds all interfaces and has no application authentication, upload
size limit, or PCAP content validation. Its exposure depends on actual network
rules; this review does not establish public reachability. The local SSH key
was not read; its permissions are `600` and Git ignores it.

Recreate the broken project virtual environment against a chosen supported
interpreter before following its activation instructions. The controller
requirements are unpinned, and runtime requirements are separate; no complete
dependency/security audit or clean-environment installation was performed.

Documentation needs reconciliation: `ProjectProgress.md:1` still says TestBed
06, and `Lite_V9_CICIoT2023/README.md:1` describes CIC-DDoS2019/TST V15 with
unrelated setup paths. `ReportSpeech.md` describes stronger capture and
completion guarantees than the current failure handling provides. Scenario
metrics use one declared class per execution, so their precision for that
class is necessarily 1 when any true positive exists; this does not measure
cross-scenario false positives.

## Reproducing the local checks

From the project root, with the system interpreter used in this review:

```bash
python3 -B -m unittest discover -s tests -v
python3 -B docs/review_2026_10_26/inference_repros.py
python3 -B docs/review_2026_10_26/pipeline_smoke.py
```

`inference_repros.py` demonstrates the three current queue/recovery failures;
it is a diagnostic reproduction, not a passing regression-test suite. Its
consumer failure cases mock model work. `pipeline_smoke.py` uses the actual
bundled model and all four inference steps. Both scripts use temporary runtime
directories and do not contact AWS. After fixes, convert the failure
reproductions into assertions for the intended recovery behavior and add
capture-shutdown and current-execution completion checks.
