# AWS IDS TestBed 06 — Project Progress

This document records the construction of `aws_ids_testbed_07` in chronological order. Historical work performed in versions 03, 04, and 05 is represented here using the final version 06 commands wherever that functionality is still present.

## Phase 1 — AWS Account and Access Preparation

1. Create and activate an Amazon AWS account.
2. Create the IAM user `aws-ids-cli-user` instead of using the AWS root user for project operations.
3. Grant the IAM user the permissions required to create, describe, tag, and terminate EC2 resources.
4. Create an access key and secret access key for the IAM user.
5. Configure the local AWS CLI:

   ```bash
   aws configure
   ```

   The project uses region `us-east-1` and JSON output.

6. Verify AWS authentication:

   ```bash
   aws sts get-caller-identity
   ```

7. Create the EC2 key pair `aws-ids-testbed-key`, save the private key as `aws-ids-testbed-key.pem`, and restrict its permissions:

   ```bash
   chmod 400 aws-ids-testbed-key.pem
   ```

8. Select the AWS network resources:

   - Region: `us-east-1`
   - VPC: `vpc-08077c2b186f60b32`
   - Subnet: `subnet-04b7fa5d91263f752`
   - Security group: `sg-08ed3dd7dcb4b98ca`

9. Configure the security group for SSH, victim HTTP traffic, internal instance communication, and the IDS receiver. Port 8000 was authorized inside the VPC with:

   ```bash
   aws ec2 authorize-security-group-ingress \
     --region us-east-1 \
     --group-id sg-08ed3dd7dcb4b98ca \
     --protocol tcp \
     --port 8000 \
     --cidr 172.31.0.0/16
   ```

## Phase 2 — Local Project Environment

1. Create the final project directory:

   ```text
   AWS_IDS_TestBed/aws_ids_testbed_07
   ```

2. Create and activate the Python environment:

   ```bash
   cd /usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_07
   python3 -m venv env_aws_ids
   source env_aws_ids/bin/activate
   ```

3. Create `requirements.txt` with `boto3`, `awscli`, `PyYAML`, `paramiko`, `scp`, `fastapi`, `uvicorn`, and `python-multipart`.
4. Install and test the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   python -c "import boto3; print('boto3 ok')"
   ```

5. Create the Python package and configuration components:

   - `aws_ids_testbed_07/__init__.py`
   - `config.yaml`: AWS, SSH, AMI, subnet, security-group, and instance settings
   - `aws_ids_testbed_07/config.py`: YAML configuration loader
   - `aws_ids_testbed_07/inventory.py`: EC2 inventory management
   - `inventory.yaml`: instance IDs, states, addresses, and DNS names
   - `aws_ids_testbed_07/ec2_lab.py`: EC2 creation, refresh, and termination
   - `aws_ids_testbed_07/remote_settings.py`: SSH key and address selection
   - `aws_ids_testbed_07/remote_runner.py`: Paramiko and SCP execution
   - `aws_ids_testbed_07/setup_service.py`: remote setup-script execution
   - `aws_ids_testbed_07/cli.py`: main command-line controller

6. Add the basic inspection commands:

   ```bash
   python -m aws_ids_testbed_07.cli status
   python -m aws_ids_testbed_07.cli show-config
   python -m aws_ids_testbed_07.cli show-inventory
   ```

## Phase 3 — Victim EC2 Instance

1. Create `scripts/setup_victim.sh`.
2. Configure it to install `nginx`, `tcpdump`, `tshark`, `curl`, `python3`, and `python3-pip`.
3. Configure it to create:

   ```text
   /opt/aws_ids_testbed/bin
   /opt/aws_ids_testbed/config
   /opt/aws_ids_testbed/csv
   /opt/aws_ids_testbed/logs
   /opt/aws_ids_testbed/pcap/writing
   /opt/aws_ids_testbed/pcap/pending
   /opt/aws_ids_testbed/pcap/sent
   /opt/aws_ids_testbed/pcap/failed
   ```

4. Configure it to create a test webpage, enable Nginx, start Nginx, and give `ubuntu` ownership of the lab folders.
5. Create, refresh, and set up the victim in that order:

   ```bash
   python -m aws_ids_testbed_07.cli create-victim
   python -m aws_ids_testbed_07.cli refresh-victim
   python -m aws_ids_testbed_07.cli setup-victim
   ```

6. Test Nginx locally on the victim:

   ```bash
   ssh -i aws-ids-testbed-key.pem ubuntu@VICTIM_PUBLIC_IP \
     "curl -s http://localhost"
   ```

## Phase 4 — IDS EC2 Instance

1. Create `scripts/setup_ids.sh`.
2. Configure it to install `python3`, `python3-pip`, `python3-venv`, `tcpdump`, `tshark`, and `curl`.
3. Configure it to create:

   ```text
   /home/ubuntu/aws_ids_testbed/input
   /home/ubuntu/aws_ids_testbed/output
   /home/ubuntu/aws_ids_testbed/models
   /home/ubuntu/aws_ids_testbed/logs
   /home/ubuntu/aws_ids_testbed/ids_env
   ```

4. Install `numpy`, `pandas`, `scikit-learn`, `joblib`, `pyyaml`, `fastapi`, `uvicorn`, and `python-multipart` in the IDS environment.
5. Create, refresh, and set up the IDS instance:

   ```bash
   python -m aws_ids_testbed_07.cli create-ids
   python -m aws_ids_testbed_07.cli refresh-ids
   python -m aws_ids_testbed_07.cli setup-ids
   ```

## Phase 5 — Attacker EC2 Instance

1. Create `scripts/setup_attacker.sh`.
2. Configure it to install:

   - `curl` for simple HTTP requests
   - `apache2-utils` for the `ab` HTTP load tool
   - `hping3` for controlled SYN traffic
   - `nmap` for scan and reconnaissance experiments
   - `python3` and `python3-pip` for future scripts

3. Create, refresh, and set up the attacker:

   ```bash
   python -m aws_ids_testbed_07.cli create-attacker
   python -m aws_ids_testbed_07.cli refresh-attacker
   python -m aws_ids_testbed_07.cli setup-attacker
   ```

4. Test attacker-to-victim communication over the victim private IP:

   ```bash
   ssh -i aws-ids-testbed-key.pem ubuntu@ATTACKER_PUBLIC_IP \
     "curl -s http://VICTIM_PRIVATE_IP"
   ```

5. Display the complete saved inventory:

   ```bash
   python -m aws_ids_testbed_07.cli show-inventory
   ```

## Phase 6 — First Manual Traffic Capture

1. Start a 20-second HTTP capture on the victim:

   ```bash
   ssh -i aws-ids-testbed-key.pem ubuntu@VICTIM_PUBLIC_IP \
     "sudo timeout 20 tcpdump -i ens5 \
     -w /opt/aws_ids_testbed/pcap/benign_http_test.pcap \
     tcp port 80"
   ```

2. While capture is running, generate benign HTTP traffic from the attacker:

   ```bash
   ssh -i aws-ids-testbed-key.pem ubuntu@ATTACKER_PUBLIC_IP \
     "ab -n 100 -c 5 http://VICTIM_PRIVATE_IP/"
   ```

3. Check the resulting PCAP file:

   ```bash
   ssh -i aws-ids-testbed-key.pem ubuntu@VICTIM_PUBLIC_IP \
     "ls -lh /opt/aws_ids_testbed/pcap/benign_http_test.pcap"
   ```

4. Preserve an example capture locally as `artifacts/pcap/benign_http_test.pcap`.

## Phase 7 — First Automated Capture and Traffic Commands

1. Create `aws_ids_testbed_07/capture_service.py` to automate the first benign capture.
2. Add capture and verification commands:

   ```bash
   python -m aws_ids_testbed_07.cli capture-benign-http
   python -m aws_ids_testbed_07.cli verify-benign-pcap
   ```

3. Create `aws_ids_testbed_07/traffic_service.py`.
4. Add the simple benign traffic command:

   ```bash
   python -m aws_ids_testbed_07.cli generate-benign-http
   ```

5. Add general traffic generation:

   ```bash
   python -m aws_ids_testbed_07.cli generate-traffic 1 \
     --requests 100 \
     --concurrency 5

   python -m aws_ids_testbed_07.cli generate-traffic 2 \
     --requests 1000 \
     --concurrency 50

   python -m aws_ids_testbed_07.cli generate-traffic 3 \
     --packet-count 1000 \
     --port 80
   ```

   Traffic codes are `1 = benign_http`, `2 = dos_http_flood`, and `3 = dos_syn_flood`.

## Phase 8 — IDS PCAP Receiver

1. Create `aws_ids_testbed_07/ids_receiver_app.py`, a FastAPI application with health-check and PCAP-upload endpoints.
2. Create `scripts/start_ids_receiver.sh` to run Uvicorn on `0.0.0.0:8000`.
3. Create `aws_ids_testbed_07/ids_receiver_service.py` to deploy, start, check, and inspect the receiver.
4. Deploy and start the receiver after completing `setup-ids`:

   ```bash
   python -m aws_ids_testbed_07.cli deploy-ids-receiver
   python -m aws_ids_testbed_07.cli ids-start-receiver
   ```

5. The receiver can also be checked directly:

   ```bash
   ssh -i aws-ids-testbed-key.pem ubuntu@IDS_PUBLIC_IP \
     "curl -s --max-time 5 http://localhost:8000/health"
   ```

6. List PCAP files received by IDS:

   ```bash
   python -m aws_ids_testbed_07.cli ids-list-received-pcaps
   ```

## Phase 9 — Configure the Attacker

1. Create `aws_ids_testbed_07/attacker_config_service.py` to generate `/opt/aws_ids_testbed/config/attacker.env` on the attacker.
2. Store the victim private IP, victim URL, request defaults, concurrency defaults, SYN packet count, and target port in that file.
3. Configure and verify the attacker target:

   ```bash
   python -m aws_ids_testbed_07.cli configure-attacker-victim-url
   python -m aws_ids_testbed_07.cli attacker-verify-victim-url
   ```

4. Create `scripts/generate_traffic.sh` for attacker-side traffic generation.
5. Create `aws_ids_testbed_07/attacker_traffic_service.py` to deploy and run the script.
6. Deploy and verify it:

   ```bash
   python -m aws_ids_testbed_07.cli deploy-attacker-traffic
   python -m aws_ids_testbed_07.cli verify-attacker-traffic
   ```

7. Run the three configured traffic types:

   ```bash
   python -m aws_ids_testbed_07.cli attacker-generate-traffic 1 \
     --requests 100 \
     --concurrency 5

   python -m aws_ids_testbed_07.cli attacker-generate-traffic 2 \
     --requests 1000 \
     --concurrency 50

   python -m aws_ids_testbed_07.cli attacker-generate-traffic 3 \
     --packet-count 100 \
     --port 80
   ```

## Phase 10 — Configure Victim-to-IDS Communication

1. Create `aws_ids_testbed_07/victim_config_service.py`.
2. Configure it to create `/opt/aws_ids_testbed/config/victim.env` containing the IDS URL and capture settings.
3. Configure and verify the victim after the IDS inventory is available:

   ```bash
   python -m aws_ids_testbed_07.cli configure-victim-ids-url
   python -m aws_ids_testbed_07.cli victim-verify-ids-url
   ```

4. Add scenario-label control:

   ```bash
   python -m aws_ids_testbed_07.cli victim-set-scenario benign_http
   python -m aws_ids_testbed_07.cli victim-set-scenario dos_http_flood
   python -m aws_ids_testbed_07.cli victim-set-scenario dos_syn_flood
   ```

## Phase 11 — General Victim PCAP Sender

1. Create `scripts/send_pcap_to_ids.sh` to upload a PCAP and its scenario label to IDS.
2. Create `aws_ids_testbed_07/victim_sender_service.py`.
3. Deploy and verify the sender:

   ```bash
   python -m aws_ids_testbed_07.cli deploy-victim-sender
   python -m aws_ids_testbed_07.cli verify-victim-sender
   ```

4. Send a specific existing PCAP:

   ```bash
   python -m aws_ids_testbed_07.cli victim-send-pcap \
     --pcap-path /opt/aws_ids_testbed/pcap/pending/benign_http_test.pcap \
     --scenario benign_http
   ```

5. Send every completed PCAP in the pending folder:

   ```bash
   python -m aws_ids_testbed_07.cli victim-send-pending-pcaps
   ```

6. Confirm receipt:

   ```bash
   python -m aws_ids_testbed_07.cli ids-list-received-pcaps
   ```

The historical `victim-send-benign-pcap` command was replaced by the general `victim-send-pcap` command.

## Phase 12 — General Scenario Capture

1. Create `scripts/capture_scenario.sh` to capture a named scenario for a requested duration.
2. Create `aws_ids_testbed_07/victim_capture_service.py`.
3. Deploy and verify the script:

   ```bash
   python -m aws_ids_testbed_07.cli deploy-victim-capture
   python -m aws_ids_testbed_07.cli verify-victim-capture
   ```

4. Capture benign HTTP traffic:

   ```bash
   # Terminal 1
   python -m aws_ids_testbed_07.cli victim-capture-scenario \
     --scenario benign_http \
     --seconds 20

   # Terminal 2
   python -m aws_ids_testbed_07.cli generate-benign-http
   ```

5. Capture controlled HTTP-flood traffic:

   ```bash
   # Terminal 1
   python -m aws_ids_testbed_07.cli victim-capture-scenario \
     --scenario dos_http_flood \
     --seconds 20

   # Terminal 2
   python -m aws_ids_testbed_07.cli generate-traffic 2 \
     --requests 1000 \
     --concurrency 50
   ```

6. Capture controlled SYN traffic:

   ```bash
   # Terminal 1
   python -m aws_ids_testbed_07.cli victim-capture-scenario \
     --scenario dos_syn_flood \
     --seconds 20

   # Terminal 2
   python -m aws_ids_testbed_07.cli generate-traffic 3 \
     --packet-count 100 \
     --port 80
   ```

7. List PCAPs in `writing`, `pending`, `sent`, and `failed`:

   ```bash
   python -m aws_ids_testbed_07.cli victim-list-pcaps
   ```

The valid command is `victim-list-pcaps`, not the older singular spelling `victim-list-pcap`.

## Phase 13 — Continuous Victim Capture Agent

1. Create `scripts/continuous_capture_agent.sh`.
2. Configure it to reload `victim.env`, read `ACTIVE_SCENARIO`, rotate HTTP captures, reject empty PCAPs, and move completed files into `pending`.
3. Create `aws_ids_testbed_07/victim_capture_agent_service.py`.
4. Add the complete capture-agent lifecycle:

   ```bash
   python -m aws_ids_testbed_07.cli deploy-victim-capture-agent
   python -m aws_ids_testbed_07.cli verify-victim-capture-agent
   python -m aws_ids_testbed_07.cli victim-start-capture-agent
   python -m aws_ids_testbed_07.cli victim-capture-agent-status
   python -m aws_ids_testbed_07.cli victim-stop-capture-agent
   ```

The valid status command is `victim-capture-agent-status`; `victim-capture-agent-stat` was a typo in earlier notes.

## Phase 14 — Automatic Pending-PCAP Sender Agent

1. Create `scripts/pending_pcap_sender_agent.sh`.
2. Configure it to monitor `pending`, extract scenario labels from filenames, upload PCAPs, move successful files to `sent`, and retain failed uploads for retry.
3. Create `aws_ids_testbed_07/victim_sender_agent_service.py`.
4. Ensure the one-file sender is deployed because the agent calls it:

   ```bash
   python -m aws_ids_testbed_07.cli deploy-victim-sender
   ```

5. Add the complete sender-agent lifecycle:

   ```bash
   python -m aws_ids_testbed_07.cli deploy-victim-sender-agent
   python -m aws_ids_testbed_07.cli verify-victim-sender-agent
   python -m aws_ids_testbed_07.cli victim-start-sender-agent
   python -m aws_ids_testbed_07.cli victim-sender-agent-status
   python -m aws_ids_testbed_07.cli victim-stop-sender-agent
   ```

## Phase 15 — Combined Agent Management

1. Add one command to start both agents:

   ```bash
   python -m aws_ids_testbed_07.cli victim-start-agents
   ```

2. Add individual and combined status commands:

   ```bash
   python -m aws_ids_testbed_07.cli victim-capture-agent-status
   python -m aws_ids_testbed_07.cli victim-sender-agent-status
   python -m aws_ids_testbed_07.cli victim-agents-status
   ```

3. Add one command that stops the sender and then the capture agent:

   ```bash
   python -m aws_ids_testbed_07.cli victim-stop-agents
   ```

## Phase 16 — One-Command Scenario Orchestration

1. Add the `run-scenario` workflow to `aws_ids_testbed_07/cli.py`.
2. Configure it to set `ACTIVE_SCENARIO`, start both victim agents, run the corresponding attacker traffic, optionally wait, and optionally stop both agents.
3. Run benign HTTP:

   ```bash
   python -m aws_ids_testbed_07.cli run-scenario 1 \
     --requests 100 \
     --concurrency 5 \
     --auto-stop-seconds 15
   ```

4. Run controlled HTTP-flood traffic:

   ```bash
   python -m aws_ids_testbed_07.cli run-scenario 2 \
     --requests 1000 \
     --concurrency 50 \
     --auto-stop-seconds 15
   ```

5. Run controlled SYN traffic:

   ```bash
   python -m aws_ids_testbed_07.cli run-scenario 3 \
     --packet-count 100 \
     --port 80 \
     --auto-stop-seconds 15
   ```

6. Inspect both ends of the pipeline:

   ```bash
   python -m aws_ids_testbed_07.cli victim-list-pcaps
   python -m aws_ids_testbed_07.cli ids-list-received-pcaps
   ```

7. If auto-stop was not supplied, stop the agents manually:

   ```bash
   python -m aws_ids_testbed_07.cli victim-stop-agents
   ```

## Phase 17 — EC2 Cleanup

1. Terminate all three roles when a test cycle is finished:

   ```bash
   python -m aws_ids_testbed_07.cli terminate-victim
   python -m aws_ids_testbed_07.cli terminate-attacker
   python -m aws_ids_testbed_07.cli terminate-ids
   ```

2. Refresh the inventory to record the final AWS states:

   ```bash
   python -m aws_ids_testbed_07.cli refresh-victim
   python -m aws_ids_testbed_07.cli refresh-attacker
   python -m aws_ids_testbed_07.cli refresh-ids
   ```

3. Display the final inventory:

   ```bash
   python -m aws_ids_testbed_07.cli show-inventory
   ```

## Final Recommended Execution Order

The following is the clean execution sequence for the final version 06 infrastructure.

```bash
cd /usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_07
source env_aws_ids/bin/activate
python -m pip install -r requirements.txt

python -m aws_ids_testbed_07.cli status
python -m aws_ids_testbed_07.cli show-config

# Victim
python -m aws_ids_testbed_07.cli create-victim
python -m aws_ids_testbed_07.cli refresh-victim
python -m aws_ids_testbed_07.cli setup-victim

# IDS
python -m aws_ids_testbed_07.cli create-ids
python -m aws_ids_testbed_07.cli refresh-ids
python -m aws_ids_testbed_07.cli setup-ids
python -m aws_ids_testbed_07.cli deploy-ids-receiver
python -m aws_ids_testbed_07.cli ids-start-receiver

# Attacker
python -m aws_ids_testbed_07.cli create-attacker
python -m aws_ids_testbed_07.cli refresh-attacker
python -m aws_ids_testbed_07.cli setup-attacker

python -m aws_ids_testbed_07.cli show-inventory

# Configure and deploy attacker traffic tools
python -m aws_ids_testbed_07.cli configure-attacker-victim-url
python -m aws_ids_testbed_07.cli attacker-verify-victim-url
python -m aws_ids_testbed_07.cli deploy-attacker-traffic
python -m aws_ids_testbed_07.cli verify-attacker-traffic

# Configure victim-to-IDS communication
python -m aws_ids_testbed_07.cli configure-victim-ids-url
python -m aws_ids_testbed_07.cli victim-verify-ids-url

# Deploy the victim sender and one-scenario capture tools
python -m aws_ids_testbed_07.cli deploy-victim-sender
python -m aws_ids_testbed_07.cli verify-victim-sender
python -m aws_ids_testbed_07.cli deploy-victim-capture
python -m aws_ids_testbed_07.cli verify-victim-capture

# Deploy continuous agents
python -m aws_ids_testbed_07.cli deploy-victim-capture-agent
python -m aws_ids_testbed_07.cli verify-victim-capture-agent
python -m aws_ids_testbed_07.cli deploy-victim-sender-agent
python -m aws_ids_testbed_07.cli verify-victim-sender-agent

# Run a complete benign scenario
python -m aws_ids_testbed_07.cli run-scenario 1 \
  --requests 100 \
  --concurrency 5 \
  --auto-stop-seconds 15

# Inspect the result
python -m aws_ids_testbed_07.cli victim-agents-status
python -m aws_ids_testbed_07.cli victim-list-pcaps
python -m aws_ids_testbed_07.cli ids-list-received-pcaps
```

The current completed pipeline is:

```text
AWS resources
→ three EC2 roles
→ role-specific setup
→ controlled attacker traffic
→ victim packet capture
→ automatic PCAP transfer
→ IDS receiver
→ stored PCAP files
```

PCAP-to-CSV conversion, trained-model deployment, inference, classification, and alert presentation remain later project stages and are not implemented by the current `aws_ids_testbed_07` code.
