"""Configuration for AWS IDS Lite V6 inference.

This config mirrors the role of Lite_V6_CICIoT2023/config/config.py, but it is
for inference on AWS IDS converted CSV files instead of training on the original
CICIoT2023 dataset.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LITE_V6_ROOT = Path(__file__).resolve().parents[1]

# Local Slurm inputs and outputs for the first inference test.
IDS_CSV_INPUT_DIR = PROJECT_ROOT / "artifacts" / "ids_csv"
IDS_INFERENCE_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ids_inference"

STEP_1_LOADED_DATA_PATH = IDS_INFERENCE_OUTPUT_DIR / "step_1_Loaded_IDS_Data.pkl"
STEP_2_PROCESSED_DATA_PATH = (
    IDS_INFERENCE_OUTPUT_DIR / "step_2_Processed_IDS_Data.pkl"
)
STEP_3_WINDOW_DATA_PATH = IDS_INFERENCE_OUTPUT_DIR / "X_IDS_windows.npy"
STEP_3_WINDOW_METADATA_PATH = IDS_INFERENCE_OUTPUT_DIR / "ids_window_metadata.csv"
STEP_3_FEATURES_USED_PATH = IDS_INFERENCE_OUTPUT_DIR / "ids_selected_features_used.pkl"
STEP_3_WINDOWING_SUMMARY_PATH = IDS_INFERENCE_OUTPUT_DIR / "ids_windowing_summary.pkl"
IDS_PREDICTIONS_PATH = IDS_INFERENCE_OUTPUT_DIR / "ids_predictions.csv"
IDS_PREDICTION_SUMMARY_PATH = IDS_INFERENCE_OUTPUT_DIR / "ids_prediction_summary.pkl"

IDS_PREDICTION_BATCH_SIZE = 64
IDS_PREDICTION_DEVICE = "auto"
IDS_CAPTURE_MANIFEST_PATH = IDS_INFERENCE_OUTPUT_DIR / "ids_capture_manifest.csv"

# Saved Lite V6 training artifacts. These are loaded, not re-created.
LITE_V6_TRAINED_PROJECT_ROOT = Path(
    "/usagers3/sanazb/Projects/LiteTransformer/Lite_V6_CICIoT2023"
)
LITE_V6_TRAINED_OUTPUTS_DIR = LITE_V6_TRAINED_PROJECT_ROOT / "outputs"

LITE_V6_MODEL_PATH = LITE_V6_TRAINED_OUTPUTS_DIR / "models" / "best_model.pt"
LITE_V6_MODEL_CONFIG_PATH = (
    LITE_V6_TRAINED_OUTPUTS_DIR / "models" / "model_config.json"
)
LITE_V6_LABEL_ENCODER_PATH = (
    LITE_V6_TRAINED_OUTPUTS_DIR / "models" / "label_encoder.pkl"
)

LITE_V6_SELECTED_FEATURES_PATH = (
    LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "selected_features.pkl"
)
LITE_V6_STANDARD_SCALER_PATH = (
    LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "standard_scaler.pkl"
)
LITE_V6_WINDOW_CONFIG_PATH = (
    LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "window_config.pkl"
)
LITE_V6_LABEL_CLASSES_PATH = (
    LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "label_classes.pkl"
)

LITE_V6_X_TRAIN_PATH = LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "X_train.npy"
LITE_V6_X_VAL_PATH = LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "X_val.npy"
LITE_V6_X_TEST_PATH = LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "X_test.npy"

LITE_V6_Y_TRAIN_PATH = LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "y_train.npy"
LITE_V6_Y_VAL_PATH = LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "y_val.npy"
LITE_V6_Y_TEST_PATH = LITE_V6_TRAINED_OUTPUTS_DIR / "data" / "y_test.npy"

IDS_DEBUG_COMPARE_LABELS = (
    "Benign_Final",
    "DoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "Recon-OSScan",
    "Recon-PortScan",
)

IDS_DEBUG_COMPARE_SAMPLE_WINDOWS_PER_LABEL = 2000

# Diagnostic-only option for live AWS inference.
# When enabled in STEP=3, these features are set to the Lite V6 scaler mean
# before scaling, so their scaled value becomes 0.
IDS_NEUTRALIZE_TIME_LEAKAGE_FEATURES = True

IDS_TIME_LEAKAGE_FEATURES = (
    "min_duration",
    "max_duration",
    "average_duration",
    "sum_duration",
    "idle_time",
)

# Development-only expected labels inferred from AWS testbed filenames.
SCENARIO_TO_EXPECTED_LABEL = {
    "benign_http": "Benign_Final",
    "dos_http_flood": "DoS-HTTP_Flood",
    "dos_syn_flood": "DoS-SYN_Flood",
}

IDS_TIMESTAMP_COLUMNS = (
    "Timestamp_Start",
    "Timestamp_End",
    "Timestamp_Mean",
)

SOURCE_TIMESTAMP_COLUMN = "Timestamp_Start"
TARGET_TIMESTAMP_COLUMN = "Timestamp"
SOURCE_LABEL_COLUMN = "expected_label_name"
TARGET_LABEL_COLUMN = "Label"

COLUMNS_TO_DROP = [
    "Source IP",
    "Destination IP",
    "Flow ID",
    "SimillarHTTP",
    "Unnamed: 0",
    "Timestamp_End",
    "Timestamp_Mean",
]
