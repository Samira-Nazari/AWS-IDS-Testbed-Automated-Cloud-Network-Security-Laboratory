#!/bin/bash
#SBATCH --job-name=lite_v9_ciciot2023_step
#SBATCH --output=outputs/logs/lite_v9_ciciot2023_step_%x_%j.out
#SBATCH --error=outputs/logs/lite_v9_ciciot2023_step_%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=40

# Lite_V9_CICIoT2023 staged runner.
#
# Run from this project directory with one of:
#   sbatch --export=ALL,STEP=1 submit_lite_v9_ciciot2023_standalone.slurm
#   sbatch --export=ALL,STEP=2 submit_lite_v9_ciciot2023_standalone.slurm
#   sbatch --export=ALL,STEP=3 submit_lite_v9_ciciot2023_standalone.slurm
#   sbatch --export=ALL,STEP=model submit_lite_v9_ciciot2023_standalone.slurm
#   sbatch --export=ALL,STEP=train submit_lite_v9_ciciot2023_standalone.slurm
#   sbatch --export=ALL,STEP=eval submit_lite_v9_ciciot2023_standalone.slurm
#
# STEP=1     load CSV files
# STEP=2     preprocess loaded data
# STEP=3     feature selection, scaling, and sliding windows
# STEP=model create Lite encoder classifier and run a forward sanity check
# STEP=train train Lite_V9_CICIoT2023
# STEP=eval  evaluate Lite_V9_CICIoT2023

set -euo pipefail

PROJECT_DIR="/usagers3/sanazb/Projects/AWS_IDS_TestBed/aws_ids_testbed_09/Lite_V9_CICIoT2023"
cd "$PROJECT_DIR"

mkdir -p outputs/logs outputs/models outputs/results outputs/data

STEP="${1:-${STEP:-1}}"
case "$STEP" in
    1|2|3|model|train|eval) ;;
    smote_enn)
        echo "STEP=smote_enn is disabled for Lite_V9_CICIoT2023."
        echo "Lite_V9_CICIoT2023 uses window-level SMOTE inside features/feature_engineering.py after valid train windows are created."
        echo "Use STEP=1, STEP=2, STEP=3, STEP=model, STEP=train, or STEP=eval."
        exit 1
        ;;
    *)
        echo "Unknown STEP=$STEP"
        echo "Use STEP=1, STEP=2, STEP=3, STEP=model, STEP=train, or STEP=eval"
        echo "Examples:"
        echo "  sbatch --export=ALL,STEP=1 submit_lite_v9_ciciot2023_standalone.slurm"
        echo "  sbatch --export=ALL,STEP=3 submit_lite_v9_ciciot2023_standalone.slurm"
        echo "  sbatch --export=ALL,STEP=model submit_lite_v9_ciciot2023_standalone.slurm"
        echo "  sbatch --export=ALL,STEP=train submit_lite_v9_ciciot2023_standalone.slurm"
        echo "  sbatch --export=ALL,STEP=eval submit_lite_v9_ciciot2023_standalone.slurm"
        exit 1
        ;;
esac

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXECUTION_LOG="outputs/logs/lite_v9_ciciot2023_step_${STEP}_${SLURM_JOB_ID:-manual}_${TIMESTAMP}.txt"

exec > >(tee -a "$EXECUTION_LOG") 2>&1

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project: $PROJECT_DIR"
echo "Step: $STEP"
echo "Execution log: $PROJECT_DIR/$EXECUTION_LOG"

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Python executable: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "Checking PyTorch CUDA visibility:"
"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

case "$STEP" in
    1)
        echo "Running STEP 1: data/data_loader.py"
        "$PYTHON_BIN" -u data/data_loader.py
        ;;
    2)
        echo "Running STEP 2: preprocessing/preprocessor.py"
        "$PYTHON_BIN" -u preprocessing/preprocessor.py
        ;;
    3)
        echo "Running STEP 3: features/feature_engineering.py"
        "$PYTHON_BIN" -u features/feature_engineering.py
        ;;
    model)
        echo "Running MODEL SANITY CHECK: models/model_config.py"
        "$PYTHON_BIN" -u models/model_config.py
        ;;
    train)
        echo "Running TRAINING: training/trainer.py"
        "$PYTHON_BIN" -u training/trainer.py
        ;;
    eval)
        echo "Running EVALUATION: evaluation/evaluator.py"
        "$PYTHON_BIN" -u evaluation/evaluator.py
        ;;
    *)
        echo "Unknown STEP=$STEP"
        echo "Use STEP=1, STEP=2, STEP=3, STEP=model, STEP=train, or STEP=eval"
        exit 1
        ;;
esac

echo "End time: $(date)"
