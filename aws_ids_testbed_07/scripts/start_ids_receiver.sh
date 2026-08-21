#!/usr/bin/env bash

# start_ids_receiver.sh
# This script starts the FastAPI PCAP receiver on the IDS EC2 instance.
# It assumes ids_receiver_app.py exists in:
#   /home/ubuntu/aws_ids_testbed/ids_receiver_app.py

set -euo pipefail

echo "[ids] Starting IDS receiver on port 8000..."

cd "$HOME/aws_ids_testbed"

"$HOME/aws_ids_testbed/ids_env/bin/python" -m uvicorn \
  ids_receiver_app:app \
  --host 0.0.0.0 \
  --port 8000
