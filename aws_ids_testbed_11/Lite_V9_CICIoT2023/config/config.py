"""
Configuration file for the CICIoT2023 Lite_V9 model run.
Contains all hyperparameters and settings for the project.
"""

import os
from pathlib import Path

import torch

# ==================== PROJECT IDENTITY ====================
PROJECT_NAME = "Lite_V9_CICIoT2023"
PROJECT_DISPLAY_NAME = "CICIoT2023 Lite_V9 Attack Detection"
EXECUTION_LOG_NAME = "lite_v9_ciciot2023_execution_process.txt"

# ==================== PROJECT PATHS ====================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path("/usagers3/sanazb/Data/CICIoT2023/CSV_files_with_labels")
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "outputs" / "models"
RESULTS_DIR = BASE_DIR / "outputs" / "results"
LOGS_DIR = BASE_DIR / "outputs" / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATASET SETTINGS ====================
DATASET_NAME = "CICIoT2023"
RUN_LABEL = "CICIoT2023/Lite_V9"
DISCOVER_CSV_FILES = True
SOURCE_LABEL_COLUMN = "label"
TARGET_LABEL_COLUMN = "Label"
SOURCE_TIMESTAMP_COLUMN = "Timestamp_Start"
TARGET_TIMESTAMP_COLUMN = "Timestamp"

# ==================== KAGGLE SETTINGS ====================
# CICIoT2023 is read from the local CSV_files_with_labels directory.
KAGGLE_DATASET_NAME = ""
KAGGLE_DATASET_PATH = str(DATA_DIR)

# ==================== DATA LOADING SETTINGS ====================
# CSV files to use from the dataset
CSV_PROTOCOLS = {
    "DrDoS_NTP": "01-12/DrDoS_NTP.csv",  # Adjust path based on your Kaggle dataset structure
    "DrDoS_DNS": "01-12/DrDoS_DNS.csv",
    "DrDoS_MSSQL": "01-12/DrDoS_MSSQL.csv",
    "DrDoS_UDP": "01-12/DrDoS_UDP.csv",
}
# ==================== ALL CSV FILES ====================
ALL_CSV_FILES = {
    # 01-12 folder
    "DrDoS_DNS": "01-12/DrDoS_DNS.csv",
    "DrDoS_LDAP": "01-12/DrDoS_LDAP.csv",
    "DrDoS_MSSQL": "01-12/DrDoS_MSSQL.csv",
    "DrDoS_NTP": "01-12/DrDoS_NTP.csv",
    "DrDoS_NetBIOS": "01-12/DrDoS_NetBIOS.csv",
    "DrDoS_SNMP": "01-12/DrDoS_SNMP.csv",
    "DrDoS_SSDP": "01-12/DrDoS_SSDP.csv",
    "DrDoS_UDP": "01-12/DrDoS_UDP.csv",
    "Syn_01_12": "01-12/Syn.csv",
    "TFTP_01_12": "01-12/TFTP.csv",
    "UDPLag_01_12": "01-12/UDPLag.csv",

    # 03-11 folder
    "LDAP_03_11": "03-11/LDAP.csv",
    "MSSQL_03_11": "03-11/MSSQL.csv",
    "NetBIOS_03_11": "03-11/NetBIOS.csv",
    "Portmap_03_11": "03-11/Portmap.csv",
    "Syn_03_11": "03-11/Syn.csv",
    "UDP_03_11": "03-11/UDP.csv",
    "UDPLag_03_11": "03-11/UDPLag.csv",
}

# Choose which CSV set the pipeline should use.
# True  -> use every file listed in ALL_CSV_FILES
# False -> use only the smaller CSV_PROTOCOLS subset
USE_ALL_CSV_FILES = True
ACTIVE_CSV_FILES = ALL_CSV_FILES if USE_ALL_CSV_FILES else CSV_PROTOCOLS

# ==================== DATA PREPROCESSING SETTINGS ====================
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
COLUMNS_TO_DROP = [
    'Source IP',
    'Destination IP',
    'Flow ID',
    'SimillarHTTP',
    'Unnamed: 0',
    'Timestamp_End',
    'Timestamp_Mean',
]

BENIGN_LABEL = 'BenignTraffic'
RANDOM_STATE = 42
BALANCE_CLASSES = False  # Keep raw class counts; handle imbalance during training
APPLY_MAX_SAMPLES_PER_LABEL = True  # If False, keep all available samples and ignore MAX_SAMPLES_PER_LABEL
MAX_SAMPLES_PER_LABEL = 1100000  # Chronologically keep at most this many rows per label
NORMALIZE_LABELS = {}

# ==================== CAPTURE SPLIT POLICY SETTINGS ====================
# legacy_capture_pools:
#   Current V9 behavior. For capped classes, train uses earliest captures,
#   test uses latest captures, and validation uses middle captures.
#
# chronological_block:
#   New requested behavior. Prefer chronological capture blocks:
#   early captures -> train, middle captures -> validation, later captures -> test.
#   Prefer different files for each split when possible. If exact 60/20/20
#   targets require splitting a boundary file, keep split rows disjoint and
#   leave an unused boundary gap.
#
# chronological_distributed_windows:
#   Window-aware behavior. First divide chronologically sorted files into
#   train/validation/test regions with enough usable windows, then choose
#   complete sliding-window chunks distributed across each region. Samples
#   inside every selected window stay consecutive in chronological order.
CAPTURE_SPLIT_POLICY = "chronological_distributed_windows"

# When a capture file must be split between train/validation/test, leave this
# many unused rows between adjacent splits. For WINDOW_SIZES=[30], 30 rows is
# the safest default because it prevents near-boundary windows from touching.
CAPTURE_SPLIT_BOUNDARY_GAP_ROWS = 30

# Prefer assigning whole capture files to only one split. Boundary splitting is
# allowed only when needed to satisfy train/validation/test target sizes.
CAPTURE_SPLIT_PREFER_FILE_EXCLUSIVE = True

# For chronological_distributed_windows, choose complete windows distributed
# across each split region instead of taking only the beginning of the region.
# "evenly_spaced" means selected window chunks are spread over the available
# chronological window starts.
WINDOW_SPLIT_SELECTION = "evenly_spaced"

# Group adjacent selected windows into row segments so the split plan stays
# compact. With WINDOW_SIZES=[30] and STEP_SIZES=[5], 512 windows correspond to
# 2,585 consecutive rows: (512 - 1) * 5 + 30.
WINDOW_SPLIT_CHUNK_WINDOWS = 512

# ==================== WINDOW-LEVEL SMOTE SETTINGS ====================
# Pre-window SMOTE is disabled because synthetic rows can break consecutive
# sliding-window semantics. Window-level SMOTE augments complete train windows
# only, after leakage-safe window creation.
APPLY_SMOTE_PRE_WINDOW = False  # Do not read step_2_Processed_Data_smote.pkl in STEP=3
APPLY_SMOTE_AFTER_WINDOW = True  # Add synthetic complete windows to train only
SMOTE_TARGET_PER_CLASS = 2000  # target number of train windows for minority classes
SMOTE_MIN_COUNT = 2000  # classes with train-window count < this are minority
SMOTE_K_NEIGHBORS = 3  # k_neighbors for window-level SMOTE
SMOTE_ENN_NEIGHBORS = 5  # neighbors to use for ENN-like synthetic cleaning
SMOTE_ENN_MIN_AGREEMENT = 1.0  # require 5 of 5 neighbors to match the synthetic class
SMOTE_RANDOM_STATE = RANDOM_STATE
SMOTE_PRESERVE_ORIGINAL_MAJORITIES = True  # do not remove original majority-class samples


# ==================== FEATURE ENGINEERING SETTINGS ====================
# Original search space:
# WINDOW_SIZES = [5, 10, 20]
# STEP_SIZES = [1, 2, 5]
WINDOW_SIZES = [30]
STEP_SIZES = [5]
RF_N_ESTIMATORS = 50
RF_RANDOM_STATE = 42
CV_SPLITS = 5
USE_ALL_FEATURES = True  # If True, skip RandomForest feature dropping after choosing the window
RF_MAX_WINDOWS_PER_LABEL = 20000  # Balanced per-label subset for RandomForest window search
RF_N_JOBS = 1  # Use a single thread for RF training to reduce memory usage

# ==================== DATA SPLITTING SETTINGS ====================
TEST_SIZE = 0.2
VALID_SIZE = 0.2
STRATIFY = True
SHUFFLE = True

# ==================== SLIDING WINDOW SETTINGS ====================
SLIDING_WINDOW_LEN = 10
SLIDING_WINDOW_STRIDE = 1

# ==================== LITE TRANSFORMER MODEL SETTINGS ====================
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128

LITE_MODEL_NAME = "LiteTransformerEncoderClassifier"
LITE_D_MODEL = 64
LITE_N_HEADS = 4
LITE_D_K = LITE_D_V = LITE_D_MODEL // LITE_N_HEADS
LITE_D_FF = 384
LITE_DROPOUT = 0.2
LITE_ACTIVATION = "gelu"
LITE_N_LAYERS = 3
LITE_FC_DROPOUT = 0.2

# Long-short range split.
# 0.5 means half of hidden channels use attention and half use convolution.
LITE_ATTENTION_FRACTION = 0.5

# Short-range convolution branch.
LITE_USE_SHORT_CONV_BRANCH = True
LITE_CONV_KERNEL_SIZE = 5
LITE_CONV_NUM_LAYERS = 2
LITE_FUSION_TYPE = "concat"

# Standalone model sanity-check fallback dimensions.
# Used only by STEP=model when X_train.npy/model_config.json do not exist yet.
LITE_SANITY_C_IN = 74
LITE_SANITY_SEQ_LEN = 30
LITE_SANITY_C_OUT = 34

if LITE_D_MODEL % LITE_N_HEADS != 0:
    raise ValueError("LITE_D_MODEL must be divisible by LITE_N_HEADS.")
if not 0.0 < LITE_ATTENTION_FRACTION < 1.0:
    raise ValueError("LITE_ATTENTION_FRACTION must be between 0 and 1.")
if LITE_CONV_KERNEL_SIZE % 2 == 0:
    raise ValueError("LITE_CONV_KERNEL_SIZE must be odd.")

# Backward-compatible aliases used by the current pipeline files.
# Step 5 will switch callers to the Lite names directly.
D_MODEL = LITE_D_MODEL
N_HEADS = LITE_N_HEADS
D_K = LITE_D_K
D_V = LITE_D_V
D_FF = LITE_D_FF
DROPOUT = LITE_DROPOUT
ACTIVATION = LITE_ACTIVATION
N_LAYERS = LITE_N_LAYERS
FC_DROPOUT = LITE_FC_DROPOUT
USE_CNN_LOCAL_BRANCH = LITE_USE_SHORT_CONV_BRANCH
CONV_KERNEL_SIZE = LITE_CONV_KERNEL_SIZE
CNN_NUM_CONV_LAYERS = LITE_CONV_NUM_LAYERS
FUSION_TYPE = LITE_FUSION_TYPE

# ==================== TRAINING SETTINGS ====================
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 12
USE_WEIGHTED_RANDOM_SAMPLER = True
SAMPLER_WEIGHT_POWER = 0.35
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

# ==================== LOSS SETTINGS ====================
LOSS_TYPE = "focal"  # Options: "weighted_ce", "focal"
FOCAL_GAMMA = 1.0

# LR Finder settings (optional)
LR_FINDER_NUM_IT = 100  # Number of iterations for LR finder

# ==================== LOGGING & CHECKPOINTING ====================
LOG_INTERVAL = 10  # Log metrics every N batches
SAVE_MODEL_INTERVAL = 1  # Save model every N epochs
VERBOSE = True

# ==================== EVALUATION SETTINGS ====================
COMPUTE_CM = True  # Compute confusion matrix
COMPUTE_ROC = True  # Compute ROC-AUC
COMPUTE_PR = True  # Compute precision-recall

# ==================== DEVICE SETTINGS ====================
USE_GPU = True
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # Number of workers for data loading (set to 0 for Windows/stable systems)

# ==================== OUTPUT SETTINGS ====================
SAVE_PREDICTIONS = True
SAVE_METRICS = True
SAVE_MODEL = True
MODEL_NAME = "lite_v9_ciciot2023_attack_detector"

print(f"✓ Configuration loaded successfully")
print(f"  - Base directory: {BASE_DIR}")
print(f"  - Data directory: {DATA_DIR}")
print(f"  - Outputs directory: {RESULTS_DIR}")
print(f"  - Device: {DEVICE}")
if USE_GPU and DEVICE == "cpu":
    print("  - CUDA requested but not available to PyTorch; falling back to CPU")
