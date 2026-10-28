#!/bin/bash

set -euo pipefail

PROJECT_DIR="/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_09/Lite_V9_CICIoT2023"
cd "$PROJECT_DIR"

mkdir -p outputs/logs outputs/models outputs/results outputs/data

EXECUTION_LOG="outputs/logs/lite_v9_ciciot2023_execution_process.txt"

# Keep normal terminal output, and copy the same output to the txt file.
exec > >(tee -a "$EXECUTION_LOG") 2>&1

echo
echo "================================================================================"
echo "Starting Lite_V9_CICIoT2023 standalone execution"
echo "Time: $(date)"
echo "Project: $PROJECT_DIR"
echo "Terminal output is also being copied to: $PROJECT_DIR/$EXECUTION_LOG"
echo "Stages: STEP=1, STEP=2, STEP=3, STEP=model, STEP=train, STEP=eval"
echo "================================================================================"

echo
echo "Python executable: $(which python)"
python --version

echo
echo "Checking CUDA availability"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "================================================================================"
echo "STEP=1: data/data_loader.py"
echo "================================================================================"
python -u data/data_loader.py

echo
echo "================================================================================"
echo "STEP=2: preprocessing/preprocessor.py"
echo "================================================================================"
python -u preprocessing/preprocessor.py

echo
echo "================================================================================"
echo "STEP=3: features/feature_engineering.py"
echo "================================================================================"
python -u features/feature_engineering.py

echo
echo "================================================================================"
echo "STEP=model: models/model_config.py"
echo "================================================================================"
python -u models/model_config.py

echo
echo "================================================================================"
echo "STEP=train: training/trainer.py"
echo "================================================================================"
python -u training/trainer.py

echo
echo "================================================================================"
echo "STEP=eval: evaluation/evaluator.py"
echo "================================================================================"
python -u evaluation/evaluator.py

echo
echo "================================================================================"
echo "Finished Lite_V9_CICIoT2023 standalone execution"
echo "Time: $(date)"
echo "Execution log: $PROJECT_DIR/$EXECUTION_LOG"
echo "================================================================================"
