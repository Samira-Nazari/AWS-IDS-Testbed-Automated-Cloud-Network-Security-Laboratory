# CIC-DDoS2019 TST V15

This project trains and evaluates a Temporal Shift Transformer-style time-series classifier on the configured CIC-DDoS2019 CSV files. V15 can run either the original Transformer-only model or the newer Transformer + CNN local branch model through `config/config.py`.

The current V15 pipeline is designed to reduce data leakage:

- raw rows are split per class in chronological order before scaling/windowing
- RandomForest feature selection is fitted on the training split only
- `StandardScaler` is fitted on selected training features only
- validation and test splits are transformed separately
- model checkpoints are selected by validation macro-F1
- optional window-level SMOTE is applied to train windows only

You can run the project either as one complete pipeline or as standalone stages.

## Project Layout

```text
TST_Complete_project_V15/
├── config/
│   └── config.py                  # Paths, data choices, hyperparameters
├── data/
│   └── data_loader.py             # Loads configured CIC-DDoS2019 CSV files
├── preprocessing/
│   └── preprocessor.py            # Sorts, cleans, drops columns
├── features/
│   └── feature_engineering.py     # Leakage-safe split, feature selection, scaling, windows
├── models/
│   └── model_config.py            # TST model definition and model factory
├── training/
│   └── trainer.py                 # DataLoaders, training loop, checkpoints
├── evaluation/
│   └── evaluator.py               # Metrics, predictions, confusion matrix
├── utils/
│   └── helpers.py                 # Logging, seeding, shape/distribution helpers
├── outputs/
│   ├── data/                      # Intermediate arrays and pickle files
│   ├── models/                    # Saved model weights and label encoder
│   ├── results/                   # Evaluation JSON, CSV, plots
│   └── logs/                      # training.log
├── main.py                        # End-to-end training and evaluation pipeline
├── submit_tst_v15_standalone.slurm # SLURM staged runner
├── submit_tst_v15_full.slurm      # SLURM full-run wrapper
├── run_standalone_v15_with_log.sh # Local/SLURM standalone stage runner
├── requirements.txt
└── README.md
```

## Setup

From the project directory:

```bash
cd /usagers3/sanazb/Projects/TST/TST_Complete_project_V15
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The code uses Python, PyTorch, scikit-learn, pandas, NumPy, matplotlib, and seaborn.

## Data

Data settings live in `config/config.py`.

Current dataset:

```python
KAGGLE_DATASET_NAME = "rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files"
DATA_DIR = Path("/usagers3/sanazb/Data/ddos2019/Version 1/raw")
```

Current CSV selection:

```python
USE_ALL_CSV_FILES = True
ACTIVE_CSV_FILES = ALL_CSV_FILES if USE_ALL_CSV_FILES else CSV_PROTOCOLS
```

With `USE_ALL_CSV_FILES = True`, the pipeline uses every file listed in `ALL_CSV_FILES`. Set it to `False` to run only the smaller `CSV_PROTOCOLS` subset.

`main.py` calls `load_data(..., auto_download=False)`, so the CSV files must already exist under `DATA_DIR`.

The standalone data loader uses `auto_download=True`. To download through Kaggle, configure credentials first:

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Run Complete Pipeline

```bash
python main.py
```

This runs:

1. raw CSV loading
2. preprocessing
3. leakage-safe feature engineering
4. saving train/validation/test arrays
5. label encoding and DataLoader creation
6. model creation
7. training
8. evaluation

## Run Standalone Stages

Run these from `TST_Complete_project_V15/`:

```bash
python data/data_loader.py
python preprocessing/preprocessor.py
python features/feature_engineering.py
python training/trainer.py
python evaluation/evaluator.py
```

Or run the standalone stages in order with logging:

```bash
./run_standalone_v15_with_log.sh
```

On SLURM, use staged runs:

```bash
sbatch --export=ALL,STEP=1 submit_tst_v15_standalone.slurm
sbatch --export=ALL,STEP=2 submit_tst_v15_standalone.slurm
sbatch --export=ALL,STEP=3 submit_tst_v15_standalone.slurm
sbatch --export=ALL,STEP=train submit_tst_v15_standalone.slurm
sbatch --export=ALL,STEP=eval submit_tst_v15_standalone.slurm
```

For a full SLURM run:

```bash
sbatch submit_tst_v15_full.slurm
```

The standalone scripts depend on files produced by earlier steps:

```text
data/data_loader.py
  -> outputs/data/step_1_Loaded_Data.pkl

preprocessing/preprocessor.py
  -> outputs/data/step_2_Processed_Data.pkl

features/feature_engineering.py
  -> outputs/data/X_train.npy
  -> outputs/data/X_val.npy
  -> outputs/data/X_test.npy
  -> outputs/data/y_train.npy
  -> outputs/data/y_val.npy
  -> outputs/data/y_test.npy
  -> outputs/data/selected_features.pkl
  -> outputs/data/dropped_features.pkl
  -> outputs/data/standard_scaler.pkl
  -> outputs/data/label_classes.pkl
  -> outputs/data/window_config.pkl

training/trainer.py
  -> outputs/models/label_encoder.pkl
  -> outputs/models/model_config.json
  -> outputs/models/best_model.pt
  -> outputs/models/best_model_epoch_*.pt
  -> outputs/models/final_model.pt

evaluation/evaluator.py
  -> outputs/results/evaluation_results.json
  -> outputs/results/predictions.csv
  -> outputs/results/confusion_matrix.png
```

`utils/helpers.py` contains reusable helpers, but it is not part of the current V15 standalone stage order. Feature engineering now creates the split arrays directly.

## Pipeline Details

### 1. Data Loading

`data/data_loader.py`:

- reads the configured CIC-DDoS2019 CSV files
- strips whitespace from column names
- converts `Timestamp` to Unix time
- keeps valid labels exactly as they appear in the CSV rows, then applies configured label normalization
- optionally caps each label to one consecutive chronological block
- combines all configured CSV files
- sorts by `Timestamp`
- saves `outputs/data/step_1_Loaded_Data.pkl` when called through `load_data`

### 2. Preprocessing

`preprocessing/preprocessor.py`:

- sorts rows chronologically by `Timestamp`
- replaces infinite values with `NaN`
- fills numeric missing values with `0`
- drops configured columns from `COLUMNS_TO_DROP`
- preserves `Label`, `unique_id`, and `Timestamp` when present
- skips standardization in the current pipeline so scaling can happen after the train/validation/test split

### 3. Leakage-Safe Feature Engineering

`features/feature_engineering.py` uses `engineer_features_no_leakage`:

- splits the dataframe per class in chronological order
- uses `VALID_SIZE` and `TEST_SIZE` from `config/config.py`
- maps labels using classes present in the training split
- removes identifiers and timestamp fields from model features
- tests all configured `WINDOW_SIZES` and `STEP_SIZES`
- uses `RandomForestClassifier` with `TimeSeriesSplit` on the training split only
- selects important features with `SelectFromModel`
- fits `StandardScaler` on selected training features only
- transforms validation and test features with the training scaler
- creates final 3D sliding-window arrays shaped as:

```text
(samples, window_size, selected_features)
```

### 4. Training

`training/trainer.py`:

- loads `X_train.npy`, `y_train.npy`, `X_val.npy`, and `y_val.npy`
- fits a `LabelEncoder` on training labels only
- saves `outputs/models/label_encoder.pkl`
- creates PyTorch DataLoaders
- rebuilds the TST model from the current input dimensions
- uses `USE_CNN_LOCAL_BRANCH`, `CONV_KERNEL_SIZE`, and `FUSION_TYPE` from `config/config.py`
- saves `outputs/models/model_config.json`
- trains with Adam and weighted cross entropy
- uses square-root inverse-frequency class weights
- tracks validation accuracy, macro-F1, weighted-F1, and balanced accuracy
- saves `best_model.pt` and `best_model_epoch_*.pt` when validation macro-F1 improves
- stops early after `EARLY_STOPPING_PATIENCE` epochs without macro-F1 improvement
- saves `final_model.pt`

### 5. Evaluation

`evaluation/evaluator.py`:

- loads `X_test.npy` and `y_test.npy`
- loads `label_encoder.pkl` and transforms test labels
- rebuilds the model from `model_config.json`
- loads `best_model.pt` when present, otherwise `final_model.pt`
- computes test loss, accuracy, balanced accuracy, weighted precision/recall/F1, macro precision/recall/F1, per-class metrics, and confusion matrix
- saves JSON results, CSV predictions, and a confusion matrix plot

## Model Architecture

The V15 model starts with the same shared embedding used by the previous TST model:

```text
Input window
(batch, window_size, num_features)
  -> Linear embedding
(batch, window_size, d_model)
  -> positional encoding
(batch, window_size, d_model)
```

When `USE_CNN_LOCAL_BRANCH = True`, the embedded sequence is sent through two parallel branches:

```text
Transformer branch:
  Multi-head self-attention encoder layers
  -> (batch, window_size, d_model)

CNN local branch:
  Linear(d_model -> 2*d_model)
  -> GLU
  -> Conv1D over time
  -> FC projection
  -> (batch, window_size, d_model)
```

The two outputs are concatenated and projected back:

```text
concat(transformer_output, cnn_output)
  -> Linear(2*d_model -> d_model)
  -> FFN
  -> mean pooling
  -> classifier
```

Set `USE_CNN_LOCAL_BRANCH = False` to run the original Transformer-only path.

## Current Configuration Highlights

Important values from `config/config.py`:

```python
WINDOW_SIZES = [5, 10, 20]
STEP_SIZES = [1, 2, 5]
RF_N_ESTIMATORS = 100
CV_SPLITS = 5

VALID_SIZE = 0.2
TEST_SIZE = 0.2

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128

D_MODEL = 48
N_HEADS = 4
D_FF = 256
DROPOUT = 0.1
ACTIVATION = "gelu"
N_LAYERS = 3
FC_DROPOUT = 0.1

USE_CNN_LOCAL_BRANCH = True
CONV_KERNEL_SIZE = 3
FUSION_TYPE = "concat"

EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
USE_WEIGHTED_RANDOM_SAMPLER = True
SAMPLER_WEIGHT_POWER = 0.35
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

USE_GPU = True
```

Runtime device selection uses CUDA only when both `USE_GPU=True` and `torch.cuda.is_available()` are true. The printed config value may say `cuda`, but the actual run log will show whether the script used `cuda` or `cpu`.

## Main vs Standalone Compatibility

`main.py` and the standalone scripts share the same handoff files. The practical difference is data download behavior: `main.py` uses `auto_download=False`, while standalone `data/data_loader.py` uses `auto_download=True`.

Model checkpoints must match the current architecture. If you want to evaluate an older Transformer-only checkpoint, set:

```python
USE_CNN_LOCAL_BRANCH = False
```

Then rebuild/evaluate with the matching model configuration.

## Output Files

### `outputs/data/`

```text
step_1_Loaded_Data.pkl          # produced by data loader / load_data
step_2_Processed_Data.pkl
X_train.npy
X_val.npy
X_test.npy
y_train.npy
y_val.npy
y_test.npy
selected_features.pkl
dropped_features.pkl
standard_scaler.pkl
label_classes.pkl
window_config.pkl
```

### `outputs/models/`

```text
label_encoder.pkl
model_config.json
best_model.pt
best_model_epoch_*.pt
final_model.pt
```

### `outputs/results/`

```text
evaluation_results.json
predictions.csv
confusion_matrix.png
```

### `outputs/logs/`

```text
training.log
```

## Troubleshooting

### CSV Files Not Found

Check `DATA_DIR` and `ACTIVE_CSV_FILES` in `config/config.py`. `main.py` will not download missing CSVs automatically.

### Kaggle Download Fails

Check that `~/.kaggle/kaggle.json` exists and has permission `600`, then run:

```bash
kaggle datasets list -s cic-ddos2019
```

### CUDA Is Not Used

The scripts use:

```python
device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
```

If evaluation logs say `Using device: cpu`, PyTorch did not see an available CUDA device in that environment.

### CUDA Out Of Memory

Reduce batch sizes in `config/config.py`:

```python
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
```

### Force CPU

```python
USE_GPU = False
```

### Missing Intermediate Files

If running standalone scripts, run all previous stages in order. For example, `training/trainer.py` needs the split files created by `features/feature_engineering.py`.

## Quick Checks

Syntax-check the main scripts:

```bash
python -m py_compile main.py data/data_loader.py preprocessing/preprocessor.py features/feature_engineering.py models/model_config.py training/trainer.py evaluation/evaluator.py
```

Watch logs during training:

```bash
tail -f outputs/logs/training.log
```

Run a model creation check in the PyTorch environment:

```bash
python models/model_config.py
```

Standalone execution log:

```text
outputs/logs/v15_execution_process.txt
```
