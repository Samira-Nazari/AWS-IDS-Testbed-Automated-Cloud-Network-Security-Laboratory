#!/usr/bin/env bash

set -euo pipefail

IDS_BASE_DIR="${IDS_BASE_DIR:-/home/ubuntu/aws_ids_testbed}"
INFERENCE_ENV="$IDS_BASE_DIR/ids_inference_env"
REQUIREMENTS_FILE="$IDS_BASE_DIR/requirements-ids-runtime.txt"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "[ids-inference-setup] Missing requirements:"
    echo "$REQUIREMENTS_FILE"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ids-inference-setup] python3 is not installed."
    exit 1
fi

echo "[ids-inference-setup] Creating runtime directories..."

mkdir -p \
    "$IDS_BASE_DIR/inference/input" \
    "$IDS_BASE_DIR/state/ids_input_agent/ready" \
    "$IDS_BASE_DIR/state/ids_input_agent/active" \
    "$IDS_BASE_DIR/state/ids_input_agent/completed" \
    "$IDS_BASE_DIR/state/ids_input_agent/failed" \
    "$IDS_BASE_DIR/logs"

echo "[ids-inference-setup] Creating inference environment..."

python3 -m venv "$INFERENCE_ENV"

PYTHON_BIN="$INFERENCE_ENV/bin/python"

echo "[ids-inference-setup] Updating pip..."

"$PYTHON_BIN" -m pip install --upgrade pip

echo "[ids-inference-setup] Installing IDS requirements..."

"$PYTHON_BIN" -m pip install \
    --requirement "$REQUIREMENTS_FILE"

echo "[ids-inference-setup] Installing CPU PyTorch..."

"$PYTHON_BIN" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0

echo "[ids-inference-setup] Checking dependencies..."

"$PYTHON_BIN" -c \
    "import numpy, pandas, scipy, sklearn, torch, joblib, dpkt, scapy, tqdm"

"$PYTHON_BIN" -c \
    "import sklearn; assert sklearn.__version__ == '1.7.2'"

echo "[ids-inference-setup] Environment is ready."
echo "[ids-inference-setup] Python: $PYTHON_BIN"
