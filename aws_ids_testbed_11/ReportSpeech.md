# AWS IDS Testbed 11 — Project Report Speech

## Purpose of this document

This document is written as a presentation script. At a normal speaking
speed, it is intended to support approximately twenty minutes of explanation.
It describes the project as it is implemented in `aws_ids_testbed_11`, not only
the original design idea. The project is an AWS-based, repeatable laboratory
for generating traffic, capturing it, converting packets into flow features,
running the existing Lite V9 IDS model, and observing the resulting diagnosis.

---

## 1. Opening: the project goal

Good morning. This project is called the AWS IDS Testbed. Its purpose is to
provide a controlled cloud laboratory for testing an intrusion detection
system, or IDS, with both benign and attack traffic.

The central question is simple: can we generate a known traffic pattern,
capture it without losing the packet data, move that capture to an IDS
instance, convert it into the same type of flow-level features expected by the
model, and produce a diagnosis with enough metadata and timing information to
explain the result?

The testbed separates traffic generation from diagnosis. The attacker or
benign generators create traffic. The victim receives that traffic and
captures it. The IDS instance receives the PCAP files and performs the
complete inference pipeline on the IDS instance. Slurm is the control plane:
the operator runs one Python CLI from Slurm, while SSH is used for deployment,
configuration, and remote control of the EC2 machines.

This separation is important for three reasons. First, it makes the test
repeatable. Second, it prevents the diagnosis from depending on a manually
downloaded file. Third, it lets us measure the time spent in capture,
transfer, conversion, inference, and alert creation separately.

The project is a laboratory and not a production security service. It uses
declared scenarios as testbed ground truth for evaluation, but the declared
scenario does not control the model prediction. The model still receives the
flow features and makes its own prediction.

---

## 2. AWS topology and network roles

The lab uses one AWS VPC and several EC2 roles.

The victim instance is the traffic receiver. It runs services such as nginx
for HTTP and HTTPS, and, for the benign IoT scenario, DNS, MQTT, and UDP
telemetry services. It also runs `tcpdump` through the continuous capture
agent.

The attacker instance is used for the controlled attack and normal HTTP
profiles. It receives the victim private IP from the generated configuration.
Scenario 2 uses ApacheBench for a controlled HTTP flood. Scenario 3 uses
`hping3` to generate SYN packets. Scenario 1 is a normal browser-like HTTP
profile.

The IDS instance is the analysis node. It hosts the FastAPI receiver, the
PCAP-to-CSV converter, the two inference agents, the alert agent, the Lite V9
model artifacts, and the event and timing logs.

Scenario 4 adds five small benign-generator instances. Each generator has a
different traffic role: HTTPS camera-like requests, DNS and UDP telemetry,
HTTP browsing, MQTT telemetry, or ICMP and UDP heartbeat traffic.

Public IP addresses are used for SSH control from Slurm. The traffic itself
uses private IP addresses inside the VPC. The attacker communicates with the
victim over the victim private IP, and the victim uploads PCAP files to the
IDS private IP. This keeps the experimental traffic inside the AWS network;
public addresses are primarily an administration mechanism.

The infrastructure settings are defined in `config.yaml`. The current
instance inventory, including instance IDs and refreshed addresses, is stored
in `inventory.yaml`. The lifecycle commands in `ec2_lab.py` create, refresh,
and terminate instances through the AWS API.

---

## 3. Project structure

The repository is organized by responsibility.

| Area | Main contents | Responsibility |
| --- | --- | --- |
| `aws_ids_testbed_11/cli.py` | The command-line interface | Maps operator commands to local and remote actions |
| `config.py`, `inventory.py`, `ec2_lab.py` | Configuration and EC2 lifecycle | Reads settings and manages the lab instances |
| `remote_runner.py`, `remote_settings.py` | SSH helpers | Connects from Slurm to EC2 instances |
| `attacker_*`, `traffic_service.py` | Attacker control | Configures the victim target and generates traffic |
| `victim_*`, `capture_service.py` | Victim control | Configures capture, starts agents, and sends PCAPs |
| `benign_*` | Benign services and generators | Builds and exercises the five-generator IoT profile |
| `ids_receiver_app.py` | FastAPI receiver | Accepts completed PCAP uploads on IDS |
| `ids_pcap_converter_service.py` and shell agent | Conversion control | Deploys and starts PCAP-to-CSV conversion |
| `ids_input_agent.py` | Agent 1 | Selects completed CSVs, maintains continuity, and publishes queue jobs |
| `ids_inference_agent.py` | Agent 2 | Runs the existing four inference steps serially per job |
| `ids_inference_pipeline.py` | Pipeline adapter | Calls the existing Step 1, Step 2, Step 3, and Step 4 functions |
| `ids_alert_agent.py` | Alert agent | Converts new prediction rows into human-readable alert events |
| `ids_timing*.py` | Timing collection and reports | Records and exports stage timing |
| `ids_prediction_metrics.py` | Scenario metrics | Calculates accuracy, target precision, and target recall |
| `scripts/` | Remote shell agents | Capture, upload, conversion, setup, and traffic scripts |
| `ids_detector/lite_v6/` | Existing inference implementation | Loads data, preprocesses features, creates windows, and predicts |
| `Lite_V9_CICIoT2023/` | Model outputs and supporting project | Contains the saved Lite V9 model and preprocessing artifacts |

The names `lite_v6` in the detector package are compatibility names retained
by the existing inference code. The runtime model artifacts used here come
from `Lite_V9_CICIoT2023/outputs`.

---

## 4. Preparation and deployment

The normal preparation begins on Slurm. The operator creates or refreshes the
victim, attacker, IDS, and optional benign-generator instances. Each instance
is then configured with its role-specific software and directories.

The victim setup installs capture tools such as `tcpdump`, nginx, curl, and
the victim-side agents. The attacker setup installs ApacheBench, `hping3`,
nmap, and curl. The IDS setup installs Python and the receiver dependencies.

For inference, `requirements-ids-runtime.txt` pins the important runtime
packages. In particular, scikit-learn is pinned to version 1.7.2 so that the
saved scaler and the runtime environment agree. The runtime also uses NumPy,
pandas, SciPy, joblib, FastAPI, Uvicorn, dpkt, Scapy, and tqdm. CPU PyTorch is
used because the IDS instance is not assumed to have a GPU.

The command `deploy-ids-inference-runtime` builds a controlled deployment
manifest. It uploads the receiver/inference support modules, the detector
Python files, the saved model, selected-feature list, scaler, window
configuration, and label classes. It does not need to upload training arrays
or local virtual environments. The remote setup script creates the IDS Python
environment, compiles the runtime files, and performs a model-loading check.

The receiver, converter, input agent, inference agent, and alert agent are
started with the corresponding CLI commands. The current implementation uses
PID files, `nohup`, and per-agent log files for this control. The logs are
stored under `/home/ubuntu/aws_ids_testbed/logs` on the IDS instance.

---

## 5. Scenario execution from one command

The main operator command is `run-scenario`. It performs the complete
workflow rather than asking the operator to manually start every individual
agent.

The available scenario codes are:

| Code | Operational scenario | Traffic source | Ground-truth class used for metrics |
| --- | --- | --- | --- |
| `1` | `benign_http` | Attacker instance using slow browser-like requests | `Benign_Final` |
| `2` | `dos_http_flood` | Attacker instance using ApacheBench | `DoS-HTTP_Flood` |
| `3` | `dos_syn_flood` | Attacker instance using hping3 SYN packets | `DoS-SYN_Flood` |
| `4` | Five-generator benign IoT traffic | Five benign-generator instances | `Benign_Final` |

Scenario 4 keeps the operational capture name `benign_http` for compatibility
with the existing victim and converter flow, but the metrics identity is
`benign_iot_5_generators`. This distinction lets the testbed preserve its
existing file flow while recording the correct experiment name and ground
truth.

At the beginning of a scenario, the CLI checks the IDS receiver, creates a
new scenario execution ID, starts Agent 1 and Agent 2, starts the alert agent,
and starts the converter. The victim is assigned its active scenario and
starts its capture and sender agents. The attacker or benign generators then
produce traffic.

When the configured traffic period ends, the CLI stops traffic, stops packet
capture, allows pending uploads to drain, stops the sender, verifies that the
PCAP files were delivered, and writes the scenario `.end.json` marker on IDS.
The `.end.json` marker means that no more input is expected. It does not mean
that diagnosis is already complete.

The command then waits for the IDS processing lifecycle to create the matching
`.complete.json` marker. Only after conversion, Agent 1 preparation, Agent 2
inference, and alert processing have finished does the CLI verify and download
the cumulative prediction CSV and the timing reports.

---

## 6. Victim capture and PCAP transfer

The victim uses `scripts/continuous_capture_agent.sh`. It reads
`/opt/aws_ids_testbed/config/victim.env` at the beginning of every capture
cycle. This means the active scenario and capture settings can change without
rewriting the script.

The default rotation is ten seconds through `CAPTURE_ROTATE_SECONDS`. A
temporary PCAP is written first. After `tcpdump` finishes, the script checks
the file size, moves a valid file into the pending directory, and writes a
small timing metadata file beside it. That metadata includes the capture ID,
scenario, PCAP filename, capture start time, capture completion time, and file
size.

The pending sender agent scans the victim pending directory. For each complete
PCAP it calls `send_pcap_to_ids.sh`. The sender submits the PCAP, source role,
scenario, timing metadata, and upload-start timestamp to the IDS FastAPI
endpoint. On success, the file is moved from `pending` to `sent` on the
victim.

The receiver exposes `/health` and `/upload-pcap`. The upload is written to a
temporary destination and then renamed into the IDS input directory. The
receiver records the receive start and end time, file size, SHA-256 checksum,
source role, scenario, and capture metadata. It also appends a structured
`pcap_received` event to the central timing log.

The receiver currently provides the laboratory upload behavior. Network
restriction, authentication, maximum upload size, and stronger PCAP content
validation are security hardening items that should be applied before using
the design outside this controlled testbed.

---

## 7. PCAP-to-CSV conversion

The IDS PCAP-to-CSV agent watches `/home/ubuntu/aws_ids_testbed/input`. When a
new PCAP appears, it calls the deployed `pcap2csv` implementation, especially
`Generating_dataset.py` and its feature-extraction modules.

The converter performs the normal feature-engineering work: it reads packets,
groups them into flows, calculates the required communication, connectivity,
layered, dynamic, and supporting features, and writes a merged CSV. If the
converter splits a large PCAP into subfiles, it converts the subfiles, merges
their CSV summaries, and removes temporary subfiles.

For every input PCAP, the converter creates either a converted marker under
`state/pcap_to_csv/converted` or a failure marker under
`state/pcap_to_csv/failed`. Agent 1 only considers a CSV usable when its
corresponding conversion marker exists and the CSV is present and non-empty.
This prevents Agent 1 from reading a file while conversion is still running.

Conversion timing is appended to `logs/ids_timing_events.jsonl`. The event
contains the PCAP name, CSV name, conversion status, start and completion
timestamps, duration, and CSV size when available.

---

## 8. Agent 1: continuity and queue preparation

Agent 1 is `ids_input_agent.py`. It is independent from Agent 2 and can keep
preparing later CSV files while Agent 2 is diagnosing an earlier job.

Agent 1 selects completed CSVs in IDS-received order. It does not wait for a
missing or delayed capture. If a later completed file is available, it can be
selected even when an earlier expected file never arrives.

The continuity rule is central to the model. The model uses a window size of
30 flow rows and a stride of 5 rows. If the current CSV has fewer than 30
combined rows, Agent 1 stores those rows in `inference/stream_buffer.csv` and
waits for the next completed CSV. If the buffer contains 25 rows and the next
CSV contains 5 rows, the combined 30 rows can produce one window.

When there are enough rows, Agent 1 creates a combined inference CSV from the
previous retained buffer and the new CSV. It calculates the window count and
the next start offset. After a successful inference job is prepared, it keeps
the tail rows needed for continuity. With a stride of 5, the next input starts
five rows later, while the last rows of the previous input remain available to
combine with the next CSV.

Agent 1 publishes a complete queue CSV first and then publishes its JSON
manifest. This ordering means Agent 2 never sees a manifest for an incomplete
CSV. The queue uses a monotonically increasing sequence and scenario-first
filenames. State markers include queued, active, completed, and failed.

---

## 9. Agent 2: the unchanged four-step IDS pipeline

Agent 2 is `ids_inference_agent.py`. It consumes the queue in sequence. Agent
1 and Agent 2 therefore work in parallel, but the inference transactions
themselves are serialized: only one job is active at a time.

For every ready job, Agent 2 performs the existing four inference functions
through `ids_inference_pipeline.py`.

**Step 1 — Load the IDS data.** The data loader reads the combined flow CSV,
normalizes the available columns into the format expected by the detector,
and stores `step_1_Loaded_IDS_Data.pkl`.

**Step 2 — Preprocess.** The preprocessor applies the existing cleaning and
feature preparation rules and stores `step_2_Processed_IDS_Data.pkl`.

**Step 3 — Create feature windows.** The feature-engineering function applies
the saved feature selection and window configuration. It creates
`X_IDS_windows.npy`, `ids_window_metadata.csv`, the selected-feature artifact,
and the windowing summary.

**Step 4 — Predict.** The evaluator loads the saved Lite V9 model, scaler,
label classes, and model configuration. It writes the current prediction CSV
and prediction summary. The normal runtime uses batch prediction and CPU
execution when no accelerator is available.

The model architecture and prediction logic are not rewritten by the AWS
agents. The pipeline adapter calls the existing functions. A diagnostic mode
can neutralize selected duration features before scaling to avoid time leakage
from those features; this is a preprocessing diagnostic setting, not a model
architecture change.

If a normal transaction fails, Agent 2 retries it once. After the second
failure, the job is marked failed, its details remain available for diagnosis,
and processing continues with later queue jobs. A successful transaction
updates the persistent master prediction CSV and then releases the next job.

---

## 10. Predictions, alerts, and monitoring

The per-transaction prediction output is merged into:

```text
/home/ubuntu/aws_ids_testbed/inference/ids_predictions.csv
```

Each row contains a prediction ID, scenario, scenario execution ID,
transaction ID, inference CSV name, window index, predicted label, confidence,
and diagnosis timestamp. The prediction ID prevents duplicate rows when a
job is retried or an agent restarts.

The alert agent watches this cumulative CSV. It remembers which prediction IDs
it has already processed and appends alert events to:

```text
/home/ubuntu/aws_ids_testbed/logs/ids_alert_events.jsonl
```

The human-readable alert record contains the scenario, inference CSV name,
window index, predicted label, confidence, verdict, UTC detection time, and a
readable local time. `Benign_Final` is reported as `BENIGN`; every other model
class is reported as `ATTACK`.

The Slurm command `ids-watch-alerts` follows the remote alert event stream and
colors benign results green, attacks red, and other status or low-confidence
messages yellow. The background JSONL files remain structured and suitable for
later analysis.

---

## 11. Timing measurements

The timing system is observational. It does not control or block the traffic
or inference pipeline.

The central IDS timing log is:

```text
/home/ubuntu/aws_ids_testbed/logs/ids_timing_events.jsonl
```

It records capture events, receiver events, conversion events, queue events,
inference events, and alert timing events. The timing records support these
metrics:

1. **PCAP capture duration** — the time used to create one rotated PCAP chunk.
2. **PCAP transfer latency** — the interval between upload start and IDS
   receipt.
3. **PCAP conversion duration** — the time to transform the PCAP into merged
   flow features.
4. **Queue waiting latency** — the time from queue publication until Agent 2
   starts the transaction.
5. **Total inference duration** — the complete Agent 2 processing duration,
   including preparation and the four pipeline steps.
6. **Pure model inference duration** — only the model execution time, excluding
   queue waiting and the surrounding data preparation.
7. **Alert processing latency** — the time between inference completion and
   recording the alert event.
8. **End-to-end detection latency** — the time from capture start to alert
   recording.

The CLI command `ids-generate-timing-reports` creates current-execution timing
CSV files on the IDS. `ids-download-timing-reports` downloads them to
`artifacts/ids_timing/`. The reports are filtered by the current scenario
execution ID so old runs are not mixed into a new report.

---

## 12. Scenario-level metrics

After the scenario has completed, the CLI downloads the cumulative prediction
CSV to:

```text
artifacts/ids_inference/ids_predictions.csv
```

The metrics module then filters that file using the current
`scenario_execution_id`. This is important because the master CSV persists
predictions from earlier runs.

The result is appended to:

```text
artifacts/ids_inference/ids_metrics.jsonl
```

The current metrics are intentionally limited to three values:

- **Accuracy:** the fraction of windows predicted as the scenario's expected
  ground-truth class.
- **Precision for the ground-truth class:** the precision of the expected
  class among the predictions for that scenario execution.
- **Recall for the ground-truth class:** the fraction of expected-class windows
  correctly predicted.

Scenario 2 uses `DoS-HTTP_Flood`, Scenario 3 uses `DoS-SYN_Flood`, and the
benign scenarios use `Benign_Final`. Because a controlled scenario has one
declared class for all of its windows, these metrics are scenario-level
evaluation values, not a complete multi-class confusion matrix. They should
be interpreted together with the per-window labels and confidence values.

---

## 13. Completion and restart behavior

The scenario lifecycle is represented by execution-specific marker files under
`state/scenario_runs` on the IDS instance. The `.end.json` marker records that
Slurm has finished sending input. The `.complete.json` marker is created only
after the converter and inference queues have drained and the quiet period has
elapsed.

The input agent uses a quiet period, currently 120 seconds by default, after
input is finished and no pending conversion or inference work remains. This
handles the fact that the last PCAP may already be uploaded while conversion
or inference is still active.

The state markers support restart safety. A queued job is visible through its
manifest. An active marker identifies the transaction being processed. A
completed marker prevents duplicate handling, and a failed marker preserves
the failure state so it can be examined. The cumulative prediction CSV uses
prediction IDs to avoid duplicate rows.

The system does not wait for missing packet sequence numbers. If packets or
captures arrive out of order, the IDS processes what it receives. Timestamp
gaps and delayed uploads are useful observations in the logs, but they do not
block diagnosis.

---

## 14. Typical operator commands

From Slurm, after activating the project environment, the operator can use:

```bash
cd /usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_11
source env_aws_ids/bin/activate
```

Infrastructure and setup examples:

```bash
python -m aws_ids_testbed_11.cli show-inventory
python -m aws_ids_testbed_11.cli deploy-ids-receiver
python -m aws_ids_testbed_11.cli deploy-ids-pcap-converter
python -m aws_ids_testbed_11.cli deploy-ids-inference-runtime
python -m aws_ids_testbed_11.cli ids-start-receiver
python -m aws_ids_testbed_11.cli ids-start-pcap-converter-agent
python -m aws_ids_testbed_11.cli ids-start-inference-agents
python -m aws_ids_testbed_11.cli ids-start-alert-agent
```

Attack scenarios:

```bash
python -m aws_ids_testbed_11.cli run-scenario 2 \
  --requests 1000 \
  --concurrency 50 \
  --auto-stop-seconds 200

python -m aws_ids_testbed_11.cli run-scenario 3 \
  --packet-count 1000 \
  --interval-microseconds 5000 \
  --auto-stop-seconds 200
```

Five-generator benign scenario:

```bash
python -m aws_ids_testbed_11.cli run-scenario 4 \
  --traffic-duration-seconds 300 \
  --auto-stop-seconds 200
```

Monitoring and downloaded outputs:

```bash
python -m aws_ids_testbed_11.cli ids-watch-alerts
tail -f artifacts/ids_inference/ids_metrics.jsonl
ls -lh artifacts/ids_timing/
ls -lh artifacts/ids_inference/ids_predictions.csv
```

The scenario command itself waits for the complete marker and verifies the
prediction CSV. The alert watch command is optional and can run in a separate
Slurm terminal while the scenario is executing.

---

## 15. Validation and limitations

The most useful acceptance test is to run one known benign case and the two
known attack cases. For each run, verify that the receiver accepts PCAPs, the
converter creates CSVs, Agent 1 publishes continuous jobs, Agent 2 executes
all four steps, the master prediction CSV grows, alerts appear, and the timing
reports contain only the current execution.

The current design has several practical limitations to remember during a
presentation. First, the IDS instance is CPU-based, so conversion and model
latency depend on the selected EC2 size and the number of windows. Second,
Scenario 4 intentionally produces sparse, realistic benign traffic; it may
produce fewer flow rows than an attack flood. The continuity buffer therefore
waits until enough rows exist for a 30-row model window. Third, the alert file
is intended for human monitoring and is separate from the richer prediction
CSV and timing JSONL. Fourth, the receiver and agent control should receive
stronger service management and security controls before production use.

These limitations do not invalidate the laboratory. They define exactly what
the testbed is measuring and make the results easier to interpret.

---

## 16. Closing summary

To conclude, AWS IDS Testbed 11 is a complete experiment pipeline. Slurm
creates and controls the AWS lab. The victim captures traffic in rotating PCAP
chunks. The sender transfers completed captures to the IDS. The receiver
records metadata and stores the PCAP. The converter produces flow-level CSV
features. Agent 1 preserves continuity across CSV boundaries and prepares
30-row, stride-5 inference inputs. Agent 2 runs the existing four IDS steps,
one transaction at a time, while Agent 1 continues preparing later inputs.
The saved Lite V9 model produces per-window predictions. The master CSV
preserves the history, the alert agent creates readable verdicts, and the
timing system explains where the time was spent.

The most important design principle is separation of concerns. Traffic
generation, capture, transfer, conversion, inference, alerting, timing, and
evaluation are separate responsibilities connected by explicit files,
markers, queues, and execution IDs. That structure makes the project
repeatable, debuggable, and suitable for comparing benign and attack
scenarios without changing the underlying model.

Thank you.
