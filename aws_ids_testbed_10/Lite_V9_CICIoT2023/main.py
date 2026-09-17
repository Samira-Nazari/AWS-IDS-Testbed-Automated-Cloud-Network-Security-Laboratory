"""
Main entry point for the Lite_V9 CICIoT2023 training pipeline.

This file mirrors the active standalone scripts:
    python data/data_loader.py
    python preprocessing/preprocessor.py
    python features/feature_engineering.py
    python utils/helpers.py
    python training/trainer.py
    python evaluation/evaluator.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from numpy_window_dataset import NumpyWindowDataset


class TeeStream:
    """Copy terminal output to a text file while keeping normal terminal printing."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()


def capture_terminal_output():
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "lite_v9_ciciot2023_execution_process.txt"
    log_file = open(log_path, "a", buffering=1)

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    print("\n" + "=" * 80)
    print(f"Terminal output is also being copied to: {log_path}")
    print("=" * 80)


capture_terminal_output()

from config.config import *
from data.data_loader import load_data
from evaluation.evaluator import Evaluator
from features.feature_engineering import engineer_features_no_leakage
from models.model_config import create_model
from preprocessing.preprocessor import preprocess_data
from training.trainer import Trainer
from utils.helpers import (
    check_data_shapes,
    get_class_distribution,
    set_seed,
    setup_logging,
    split_per_class_time_series,
)

logger = setup_logging(str(LOGS_DIR), "training.log")


def _save_pickle(obj, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def _prepare_feature_engineering_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same result-affecting cleanup used by features/feature_engineering.py.
    """
    df = df.drop(
        columns=[
            "Source IP",
            "Destination IP",
            "SimillarHTTP",
            "Flow ID",
            "unique_id",
            "Timestamp",
        ],
        errors="ignore",
    )
    df["Label"], _ = pd.factorize(df["Label"])
    return df.select_dtypes(include=[np.number])


def _save_window_outputs(
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    selected_features: list,
    columns_to_drop: list,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_windows.npy", X_windows)
    np.save(output_dir / "y_windows.npy", y_windows)
    pd.to_pickle(selected_features, output_dir / "selected_features.pkl")
    pd.to_pickle(columns_to_drop, output_dir / "dropped_features.pkl")


def _save_split_outputs(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "y_test.npy", y_test)



def main():
    """Run the complete Lite_V9 CICIoT2023 pipeline from raw CSV files to final evaluation."""
    logger.info("\n")
    logger.info("█" * 80)
    logger.info("█" + " " * 78 + "█")
    logger.info("█" + " " * 15 + "CICIoT2023 LITE_V9 TRAINING PIPELINE" + " " * 22 + "█")
    logger.info("█" + " " * 78 + "█")
    logger.info("█" * 80)

    logger.info("\nSetting random seed...")
    set_seed(RANDOM_STATE)

    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    logger.info(f"\nUsing device: {device}")

    data_output_dir = BASE_DIR / "outputs" / "data"
    data_output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # ==================== STEP 1: LOAD DATA ====================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: LOAD DATA")
        logger.info("=" * 60)

        raw_data = load_data(
            KAGGLE_DATASET_NAME,
            str(DATA_DIR),
            ACTIVE_CSV_FILES,
            auto_download=False,
        )
        logger.info(f"\nRaw data shape: {raw_data.shape}")

        # ==================== STEP 2: PREPROCESS DATA ====================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: PREPROCESS DATA")
        logger.info("=" * 60)

        preprocessed_data = preprocess_data(
            raw_data,
            COLUMNS_TO_DROP,
            fit_scaler=True,
            standardize=False,
        )
        processed_path = data_output_dir / "step_2_Processed_Data.pkl"
        preprocessed_data.to_pickle(processed_path)
        logger.info(f"Saved processed data to: {processed_path}")
        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")

        # ==================== STEP 3: FEATURE ENGINEERING ====================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("=" * 60)

        result = engineer_features_no_leakage(
            preprocessed_data,
            WINDOW_SIZES,
            STEP_SIZES,
            target_column="Label",
            valid_size=VALID_SIZE,
            test_size=TEST_SIZE,
            rf_n_estimators=RF_N_ESTIMATORS,
            cv_splits=CV_SPLITS,
        )
        X_train = result["X_train"]
        X_val = result["X_val"]
        X_test = result["X_test"]
        y_train = result["y_train"]
        y_val = result["y_val"]
        y_test = result["y_test"]

        pd.to_pickle(result["selected_features"], data_output_dir / "selected_features.pkl")
        pd.to_pickle(result["columns_to_drop"], data_output_dir / "dropped_features.pkl")
        pd.to_pickle(result["scaler"], data_output_dir / "standard_scaler.pkl")
        pd.to_pickle(result["label_classes"], data_output_dir / "label_classes.pkl")
        result["window_segment_audit"].to_csv(
            data_output_dir / "window_segment_audit.csv", index=False
        )
        result["smote_audit"].to_csv(
            data_output_dir / "smote_audit.csv", index=False
        )
        pd.to_pickle(
            result["windowing_summary"],
            data_output_dir / "windowing_summary.pkl",
        )
        pd.to_pickle(
            {
                "best_window_size": result["best_window_size"],
                "best_step_size": result["best_step_size"],
            },
            data_output_dir / "window_config.pkl"
        )

        # ==================== STEP 4: SAVE SPLIT WINDOWS ====================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: SAVE LEAKAGE-SAFE TRAIN/VAL/TEST WINDOWS")
        logger.info("=" * 60)

        check_data_shapes(X_train, y_train)
        check_data_shapes(X_val, y_val)
        check_data_shapes(X_test, y_test)

        logger.info("\nTraining set distribution before label encoding:")
        get_class_distribution(y_train)
        logger.info("\nValidation set distribution before label encoding:")
        get_class_distribution(y_val)
        logger.info("\nTest set distribution before label encoding:")
        get_class_distribution(y_test)

        _save_split_outputs(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            data_output_dir,
        )
        logger.info(f"Saved split arrays to: {data_output_dir}")

        # ==================== STEP 5: LABEL ENCODING + DATA LOADERS ====================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: LABEL ENCODING AND DATA LOADERS")
        logger.info("=" * 60)

        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)

        # Validate that validation/test labels are known to the encoder
        train_classes = set(label_encoder.classes_)
        val_unknown = set(np.unique(y_val)) - train_classes
        test_unknown = set(np.unique(y_test)) - train_classes
        if val_unknown:
            raise ValueError(
                f"Validation contains labels not seen in training: {sorted(val_unknown)}"
            )
        if test_unknown:
            raise ValueError(
                f"Test contains labels not seen in training: {sorted(test_unknown)}"
            )
        logger.info(f"Label encoder classes: {label_encoder.classes_.tolist()}")

        y_train_encoded = label_encoder.transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)

        le_path = MODELS_DIR / "label_encoder.pkl"
        _save_pickle(label_encoder, le_path)
        logger.info(f"Label encoder saved to: {le_path}")
        logger.info(f"Encoded train labels: {np.unique(y_train_encoded)}")
        logger.info(f"Encoded validation labels: {np.unique(y_val_encoded)}")
        logger.info(f"Encoded test labels: {np.unique(y_test_encoded)}")

        trainer = Trainer(
            model=None,
            device=device,
            output_dir=str(MODELS_DIR),
        )

        train_loader, val_loader = trainer.create_dataloaders(
            X_train,
            y_train_encoded,
            X_val,
            y_val_encoded,
            batch_size=BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            use_weighted_sampler=USE_WEIGHTED_RANDOM_SAMPLER,
            sampler_weight_power=SAMPLER_WEIGHT_POWER,
        )

        # ==================== STEP 6: CREATE MODEL ====================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: CREATE MODEL")
        logger.info("=" * 60)

        c_in = X_train.shape[2]
        seq_len = X_train.shape[1]
        n_classes = len(label_encoder.classes_)

        logger.info("\nData dimensions:")
        logger.info(f"  c_in: {c_in}")
        logger.info(f"  seq_len: {seq_len}")
        logger.info(f"  n_classes: {n_classes}")

        model, model_config = create_model(
            c_in=c_in,
            c_out=n_classes,
            seq_len=seq_len,
            d_model=LITE_D_MODEL,
            n_heads=LITE_N_HEADS,
            d_ff=LITE_D_FF,
            dropout=LITE_DROPOUT,
            activation=LITE_ACTIVATION,
            n_layers=LITE_N_LAYERS,
            fc_dropout=LITE_FC_DROPOUT,
            use_cnn_local_branch=LITE_USE_SHORT_CONV_BRANCH,
            conv_kernel_size=LITE_CONV_KERNEL_SIZE,
            cnn_num_conv_layers=LITE_CONV_NUM_LAYERS,
            fusion_type=LITE_FUSION_TYPE,
            attention_fraction=LITE_ATTENTION_FRACTION,
            device=device,
        )

        config_path = MODELS_DIR / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(model_config.to_dict(), f, indent=2)
        logger.info(f"Model config saved to: {config_path}")

        # ==================== STEP 7: TRAIN MODEL ====================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 7: TRAIN MODEL")
        logger.info("=" * 60)

        trainer = Trainer(
            model=model,
            device=device,
            output_dir=str(MODELS_DIR),
        )
        history = trainer.fit(
            train_loader,
            val_loader,
            num_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            log_interval=LOG_INTERVAL,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            loss_type=LOSS_TYPE,
            focal_gamma=FOCAL_GAMMA,
        )

        best_model_path = MODELS_DIR / "best_model.pt"
        final_model_path = MODELS_DIR / "final_model.pt"
        model_path = best_model_path if best_model_path.exists() else final_model_path
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture does not match the current Lite_V9 model. "
                "Train Lite_V9 first to create a compatible checkpoint."
            ) from exc
        logger.info(f"Loaded trained model for evaluation from: {model_path}")

        # ==================== STEP 8: EVALUATE MODEL ====================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 8: EVALUATE MODEL")
        logger.info("=" * 60)

        test_dataset = NumpyWindowDataset(X_test, y_test_encoded)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

        evaluator = Evaluator(
            model=model,
            device=device,
            output_dir=str(RESULTS_DIR),
        )

        criterion = torch.nn.CrossEntropyLoss()
        results = evaluator.evaluate(test_loader, criterion)
        evaluator.save_results(results)
        evaluator.save_predictions(results)
        evaluator.plot_confusion_matrix(np.array(results["confusion_matrix"]))

        logger.info("\n\n")
        logger.info("█" * 80)
        logger.info("█" + " " * 78 + "█")
        logger.info("█" + " " * 16 + "LITE_V9 CICIoT2023 PIPELINE COMPLETED" + " " * 21 + "█")
        logger.info("█" + " " * 78 + "█")
        logger.info("█" * 80)

        logger.info("\nSummary:")
        logger.info(f"  Training samples: {len(y_train_encoded)}")
        logger.info(f"  Validation samples: {len(y_val_encoded)}")
        logger.info(f"  Test samples: {len(y_test_encoded)}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Test accuracy: {results['metrics']['accuracy']:.6f}")
        logger.info(f"  Test macro-F1: {results['metrics']['macro_f1']:.6f}")
        logger.info(f"  Models: {MODELS_DIR}")
        logger.info(f"  Results: {RESULTS_DIR}")
        logger.info(f"  Logs: {LOGS_DIR}")

        return {
            "model": model,
            "trainer": trainer,
            "evaluator": evaluator,
            "results": results,
            "history": history,
            "label_encoder": label_encoder,
            "selected_features": result["selected_features"],
            "columns_to_drop": result["columns_to_drop"],
            "dataloaders": (train_loader, val_loader, test_loader),
        }

    except Exception as e:
        logger.error("\n\nERROR DURING PIPELINE EXECUTION")
        logger.error("=" * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("=" * 80)
        logger.exception("Full traceback:")
        raise


if __name__ == '__main__':
    try:
        outputs = main()
        logger.info("\n✅ Done!")
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Pipeline failed: {e}")
        sys.exit(1)
