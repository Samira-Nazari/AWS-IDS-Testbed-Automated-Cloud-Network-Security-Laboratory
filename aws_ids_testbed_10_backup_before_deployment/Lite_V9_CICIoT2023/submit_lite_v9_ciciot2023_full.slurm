#!/bin/bash
#SBATCH --job-name=lite_v9_ciciot2023_full
#SBATCH --output=outputs/logs/lite_v9_ciciot2023_full_%j.out
#SBATCH --error=outputs/logs/lite_v9_ciciot2023_full_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=40

# Lite_V9_CICIoT2023 full-run wrapper.
#
# Full pipeline:
#   sbatch submit_lite_v9_ciciot2023_full.slurm
#
# Model-only sanity check:
#   sbatch --export=ALL,RUN_MODE=model_check submit_lite_v9_ciciot2023_full.slurm
#
# RUN_MODE=full        runs the complete standalone pipeline
# RUN_MODE=model_check runs only models/model_config.py

set -euo pipefail

PROJECT_DIR="/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_09/Lite_V9_CICIoT2023"
cd "$PROJECT_DIR"

mkdir -p outputs/logs outputs/models outputs/results outputs/data

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXECUTION_LOG="outputs/logs/execution_lite_v9_ciciot2023_full_${SLURM_JOB_ID:-manual}_${TIMESTAMP}.txt"

# Duplicate everything printed by this script and the Python pipeline into
# one readable text file while still sending it to the SLURM stdout/stderr logs.
exec > >(tee -a "$EXECUTION_LOG") 2>&1

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project: $PROJECT_DIR"
echo "Execution log: $PROJECT_DIR/$EXECUTION_LOG"

echo "GPU status:"
nvidia-smi || true

echo "Loading environment modules..."
set +e
set +u
source /etc/profile.d/modules.sh
echo "modules.sh status: $?"
echo "Loading anaconda3 module..."
module load anaconda3
echo "module load status: $?"
echo "Loading conda shell hook..."
source /usr/local/anaconda3/etc/profile.d/conda.sh
echo "conda.sh status: $?"
echo "Activating conda environment: tst_gpu"
conda activate tst_gpu
echo "conda activate status: $?"
set -e
set -u
echo "Conda environment activated"

PYTHON_BIN="/usagers3/sanazb/.conda/envs/tst_gpu/bin/python"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Python executable: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "Checking PyTorch CUDA:"
"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Enforcing GPU execution:"
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Stop job because GPU was requested.")
print("CUDA is available:", torch.cuda.get_device_name(0))
PY

RUN_MODE="${RUN_MODE:-full}"

case "$RUN_MODE" in
    model_check)
        echo "Running Lite_V9_CICIoT2023 model check: python -u models/model_config.py"
        "$PYTHON_BIN" -u models/model_config.py
        ;;
    full)
        echo "Running Lite_V9_CICIoT2023 standalone pipeline in order: ./run_standalone_lite_v9_ciciot2023_with_log.sh"
        ./run_standalone_lite_v9_ciciot2023_with_log.sh
        ;;
    *)
        echo "Unknown RUN_MODE=$RUN_MODE"
        echo "Use RUN_MODE=full or RUN_MODE=model_check"
        exit 1
        ;;
esac

echo "End time: $(date)"
