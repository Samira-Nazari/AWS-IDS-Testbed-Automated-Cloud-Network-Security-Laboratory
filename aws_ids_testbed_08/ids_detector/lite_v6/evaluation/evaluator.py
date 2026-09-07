"""
Model Evaluation module for TST
Handles evaluation and computation of various metrics
"""
############# for standalone evaluation #############
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
###############

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, auc)
import json
import logging
import pickle
from pathlib import Path

try:
    from ids_detector.lite_v6.config.config import (
        IDS_INFERENCE_OUTPUT_DIR,
        IDS_DEBUG_COMPARE_LABELS,
        IDS_DEBUG_COMPARE_SAMPLE_WINDOWS_PER_LABEL,
        IDS_PREDICTION_BATCH_SIZE,
        IDS_PREDICTION_DEVICE,
        IDS_PREDICTION_SUMMARY_PATH,
        IDS_PREDICTIONS_PATH,
        LITE_V6_LABEL_CLASSES_PATH,
        LITE_V6_MODEL_CONFIG_PATH,
        LITE_V6_MODEL_PATH,
        LITE_V6_SELECTED_FEATURES_PATH,
        LITE_V6_STANDARD_SCALER_PATH,
        LITE_V6_WINDOW_CONFIG_PATH,
        LITE_V6_X_TEST_PATH,
        LITE_V6_X_TRAIN_PATH,
        LITE_V6_X_VAL_PATH,
        LITE_V6_Y_TEST_PATH,
        LITE_V6_Y_TRAIN_PATH,
        LITE_V6_Y_VAL_PATH,
        STEP_2_PROCESSED_DATA_PATH,
        STEP_3_FEATURES_USED_PATH,
        STEP_3_WINDOW_DATA_PATH,
        STEP_3_WINDOW_METADATA_PATH,
    )
    from ids_detector.lite_v6.models.model_config import create_model
    from ids_detector.lite_v6.numpy_window_dataset import NumpyWindowDataset
except ModuleNotFoundError as exc:
    if exc.name != "ids_detector":
        raise

    from config.config import (
        IDS_INFERENCE_OUTPUT_DIR,
        IDS_DEBUG_COMPARE_LABELS,
        IDS_DEBUG_COMPARE_SAMPLE_WINDOWS_PER_LABEL,
        IDS_PREDICTION_BATCH_SIZE,
        IDS_PREDICTION_DEVICE,
        IDS_PREDICTION_SUMMARY_PATH,
        IDS_PREDICTIONS_PATH,
        LITE_V6_LABEL_CLASSES_PATH,
        LITE_V6_MODEL_CONFIG_PATH,
        LITE_V6_MODEL_PATH,
        LITE_V6_SELECTED_FEATURES_PATH,
        LITE_V6_STANDARD_SCALER_PATH,
        LITE_V6_WINDOW_CONFIG_PATH,
        LITE_V6_X_TEST_PATH,
        LITE_V6_X_TRAIN_PATH,
        LITE_V6_X_VAL_PATH,
        LITE_V6_Y_TEST_PATH,
        LITE_V6_Y_TRAIN_PATH,
        LITE_V6_Y_VAL_PATH,
        STEP_2_PROCESSED_DATA_PATH,
        STEP_3_FEATURES_USED_PATH,
        STEP_3_WINDOW_DATA_PATH,
        STEP_3_WINDOW_METADATA_PATH,
    )
    from models.model_config import create_model
    from numpy_window_dataset import NumpyWindowDataset

logger = logging.getLogger(__name__)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _resolve_ids_prediction_device(device_setting: str) -> str:
    """Resolve AWS IDS prediction device from config/CLI."""
    normalized = str(device_setting).strip().lower()

    if normalized in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if normalized == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested, but not available. Falling back to CPU.")
        return "cpu"

    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported IDS prediction device: {device_setting}")

    return normalized


def _load_lite_v6_label_names() -> list[str]:
    """Load the original Lite V6 class-name order used by model outputs."""
    label_classes = _load_pickle(LITE_V6_LABEL_CLASSES_PATH)
    return [str(label_name) for label_name in list(label_classes)]


def _load_lite_v6_trained_model(device: str) -> tuple[nn.Module, dict]:
    """Rebuild the Lite V6 model and load saved trained weights."""
    config_dict = _load_json(LITE_V6_MODEL_CONFIG_PATH)

    model, _ = create_model(
        c_in=config_dict["c_in"],
        c_out=config_dict["c_out"],
        seq_len=config_dict["seq_len"],
        d_model=config_dict["d_model"],
        n_heads=config_dict["n_heads"],
        d_ff=config_dict["d_ff"],
        dropout=config_dict["dropout"],
        activation=config_dict["activation"],
        n_layers=config_dict["n_layers"],
        fc_dropout=config_dict["fc_dropout"],
        use_cnn_local_branch=config_dict.get("use_cnn_local_branch", True),
        conv_kernel_size=config_dict.get("conv_kernel_size", 5),
        cnn_num_conv_layers=config_dict.get("cnn_num_conv_layers", 2),
        fusion_type=config_dict.get("fusion_type", "concat"),
        attention_fraction=config_dict.get("attention_fraction", 0.5),
        device=device,
    )

    try:
        state_dict = torch.load(LITE_V6_MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(LITE_V6_MODEL_PATH, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    return model, config_dict


def _safe_probability_column_name(label_name: str) -> str:
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in str(label_name).strip()
    )
    return "_".join(part for part in safe_name.split("_") if part)


def resolve_device(use_gpu: bool) -> str:
    """Choose the evaluation device and log CUDA fallback details."""
    if not use_gpu:
        return "cpu"

    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"

    logger.warning(
        "USE_GPU=True, but CUDA is not available to PyTorch. "
        "Evaluation will run on CPU. On Slurm, request a GPU allocation "
        "before launching this script."
    )
    return "cpu"


class Evaluator:
    """Evaluate TST model performance"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu',
                 output_dir: str = 'outputs/results'):
        """
        Args:
            model: Trained PyTorch model
            device: Device to evaluate on
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def predict(self, test_loader: DataLoader) -> tuple:
        """
        Get predictions on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (predictions, probabilities, labels)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)

    def predict_probabilities(self, data_loader: DataLoader) -> tuple:
        """
        Get predictions when true labels are not required.

        Args:
            data_loader: Data loader returning either x or (x, y)

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in data_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_probs: np.ndarray = None, average: str = 'weighted') -> dict:
        """
        Compute evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Predicted probabilities
            average: Averaging method for multi-class ('weighted', 'macro', 'micro')
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Weighted metrics
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

        # Macro metrics are important when classes are imbalanced.
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, 
                                                         average=None, zero_division=0).tolist()
        metrics['recall_per_class'] = recall_score(y_true, y_pred, 
                                                   average=None, zero_division=0).tolist()
        metrics['f1_per_class'] = f1_score(y_true, y_pred, 
                                          average=None, zero_division=0).tolist()
        
        # ROC-AUC (only for binary classification)
        if len(np.unique(y_true)) == 2 and y_probs is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        
        return metrics
    
    def evaluate(self, test_loader: DataLoader, criterion: nn.Module = None) -> dict:
        """
        Full evaluation on test set
        
        Args:
            test_loader: Test data loader
            criterion: Loss function (optional)
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 7: EVALUATING MODEL")
        logger.info("="*60)
        
        # Get predictions
        logger.info("\n🔮 Getting predictions...")
        y_pred, y_probs, y_true = self.predict(test_loader)
        
        # Compute loss if criterion provided
        test_loss = None
        if criterion is not None:
            self.model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    loss = criterion(logits, y)
                    total_loss += loss.item() * x.size(0)
            test_loss = total_loss / len(y_true)
        
        # Compute metrics
        logger.info("📊 Computing metrics...")
        metrics = self.compute_metrics(y_true, y_pred, y_probs)
        
        # Confusion matrix
        logger.info("📈 Computing confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        
        if test_loss is not None:
            logger.info(f"\nTest Loss: {test_loss:.6f}")
        
        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Accuracy:          {metrics['accuracy']:.6f}")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.6f}")
        logger.info(f"  Weighted Precision:{metrics['precision']:.6f}")
        logger.info(f"  Weighted Recall:   {metrics['recall']:.6f}")
        logger.info(f"  Weighted F1-Score: {metrics['f1']:.6f}")
        logger.info(f"  Macro Precision:   {metrics['macro_precision']:.6f}")
        logger.info(f"  Macro Recall:      {metrics['macro_recall']:.6f}")
        logger.info(f"  Macro F1-Score:    {metrics['macro_f1']:.6f}")
        
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.6f}")
        
        logger.info(f"\nPer-Class Metrics:")
        n_classes = len(np.unique(y_true))
        for i in range(n_classes):
            logger.info(f"  Class {i}:")
            logger.info(f"    Precision: {metrics['precision_per_class'][i]:.6f}")
            logger.info(f"    Recall:    {metrics['recall_per_class'][i]:.6f}")
            logger.info(f"    F1-Score:  {metrics['f1_per_class'][i]:.6f}")
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"{cm}")

        true_classes, true_counts = np.unique(y_true, return_counts=True)
        pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
        logger.info(f"\nTrue label distribution: {dict(zip(true_classes.astype(int), true_counts.astype(int)))}")
        logger.info(f"Pred label distribution: {dict(zip(pred_classes.astype(int), pred_counts.astype(int)))}")
        
        # Sample predictions
        logger.info(f"\nSample Predictions (first 20):")
        for i in range(min(20, len(y_true))):
            correct = "✓" if y_true[i] == y_pred[i] else "✗"
            logger.info(f"  [{correct}] True: {y_true[i]:2d} | "
                       f"Pred: {y_pred[i]:2d} | "
                       f"Conf: {y_probs[i].max():.4f}")
        
        logger.info("="*60 + "\n")
        
        results = {
            'test_loss': test_loss,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_probs.tolist(),
            'true_labels': y_true.tolist()
        }
        
        return results
    
    def save_results(self, results: dict, filename: str = 'evaluation_results.json'):
        """
        Save evaluation results to JSON
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'test_loss': float(results['test_loss']) if results['test_loss'] is not None else None,
            'metrics': {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                       for k, v in results['metrics'].items()},
            'confusion_matrix': results['confusion_matrix'],
            'summary': {
                'total_samples': len(results['true_labels']),
                'accuracy': float(results['metrics']['accuracy']),
                'balanced_accuracy': float(results['metrics']['balanced_accuracy']),
                'f1_score': float(results['metrics']['f1']),
                'macro_f1_score': float(results['metrics']['macro_f1']),
                'precision': float(results['metrics']['precision']),
                'recall': float(results['metrics']['recall'])
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"✓ Results saved to {filepath}")
    
    def save_predictions(self, results: dict, filename: str = 'predictions.csv'):
        """
        Save predictions to CSV
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        df = pd.DataFrame({
            'true_label': results['true_labels'],
            'predicted_label': results['predictions'],
            'confidence': np.max(results['probabilities'], axis=1)
        })
        
        # Add per-class probabilities
        for i in range(len(results['probabilities'][0])):
            df[f'prob_class_{i}'] = [row[i] for row in results['probabilities']]
        
        df.to_csv(filepath, index=False)
        logger.info(f"✓ Predictions saved to {filepath}")

    def save_ids_predictions(
        self,
        predictions: pd.DataFrame,
        output_path: Path = IDS_PREDICTIONS_PATH,
    ) -> None:
        """
        Save AWS IDS window-level predictions to CSV.

        Args:
            predictions: Window prediction rows with IDS metadata
            output_path: Full output CSV path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        logger.info(f"IDS predictions saved to {output_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, filename: str = 'confusion_matrix'):
        """
        Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix
            filename: Output filename (without extension)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            filepath = self.output_dir / f'{filename}.png'
            plt.savefig(filepath, dpi=300)
            plt.close()
            
            logger.info(f"✓ Confusion matrix saved to {filepath}")
        except ImportError:
            logger.warning("⚠️  Matplotlib not available for plotting")


def _validate_ids_prediction_inputs(
    windows: np.ndarray,
    metadata: pd.DataFrame,
    model_config: dict,
    label_names: list[str],
) -> None:
    """Validate that IDS windows match the trained Lite V6 model contract."""
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D window array, got shape: {windows.shape}")

    if len(windows) != len(metadata):
        raise ValueError(
            "Window data and metadata row counts do not match: "
            f"{len(windows)} != {len(metadata)}"
        )

    expected_seq_len = int(model_config["seq_len"])
    expected_c_in = int(model_config["c_in"])
    if windows.shape[1] != expected_seq_len or windows.shape[2] != expected_c_in:
        raise ValueError(
            "IDS window shape does not match Lite V6 model config: "
            f"windows={windows.shape}, expected=(*, {expected_seq_len}, {expected_c_in})"
        )

    expected_c_out = int(model_config["c_out"])
    if len(label_names) != expected_c_out:
        raise ValueError(
            "Lite V6 label-name count does not match model output count: "
            f"{len(label_names)} != {expected_c_out}"
        )


def _predict_ids_window_group(
    evaluator: Evaluator,
    windows: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict one file's windows, using smaller final batches when needed."""
    dummy_labels = np.zeros(len(windows), dtype=np.int64)
    dataset = NumpyWindowDataset(windows, dummy_labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    return evaluator.predict_probabilities(loader)


def _build_ids_prediction_rows(
    group_metadata: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """Build the saved window-level AWS IDS prediction table."""
    output = group_metadata.copy().reset_index(drop=True)
    output["predicted_label_index"] = predictions.astype(int)
    output["predicted_label_name"] = [
        label_names[int(class_index)] for class_index in predictions
    ]
    output["confidence"] = probabilities.max(axis=1)

    if {
        "start_source_row_index",
        "end_source_row_index",
    }.issubset(output.columns):
        output["start_flow_number"] = output["start_source_row_index"].astype(int) + 1
        output["end_flow_number"] = output["end_source_row_index"].astype(int) + 1
        output["flow_range"] = (
            "flows "
            + output["start_flow_number"].astype(str)
            + "-"
            + output["end_flow_number"].astype(str)
        )

    if "expected_label_name" in output.columns:
        output["expected_match"] = (
            output["expected_label_name"].astype(str)
            == output["predicted_label_name"].astype(str)
        )

    for class_index, label_name in enumerate(label_names):
        safe_label = _safe_probability_column_name(label_name)
        output[f"prob_{class_index}_{safe_label}"] = probabilities[:, class_index]

    preferred_columns = [
        "window_index",
        "source_file_name",
        "local_window_index",
        "flow_range",
        "start_flow_number",
        "end_flow_number",
        "start_source_row_index",
        "end_source_row_index",
        "start_timestamp",
        "end_timestamp",
        "window_size",
        "step_size",
        "expected_scenario",
        "expected_label_name",
        "predicted_label_index",
        "predicted_label_name",
        "confidence",
        "expected_match",
    ]
    ordered_columns = [
        column for column in preferred_columns if column in output.columns
    ]
    remaining_columns = [
        column for column in output.columns if column not in ordered_columns
    ]
    return output[ordered_columns + remaining_columns]


def predict_ids_windows(
    window_data_path: Path = STEP_3_WINDOW_DATA_PATH,
    metadata_path: Path = STEP_3_WINDOW_METADATA_PATH,
    predictions_output_path: Path = IDS_PREDICTIONS_PATH,
    summary_output_path: Path = IDS_PREDICTION_SUMMARY_PATH,
    batch_size: int = IDS_PREDICTION_BATCH_SIZE,
    device_setting: str = IDS_PREDICTION_DEVICE,
) -> dict:
    """Run Lite V6 prediction on AWS IDS windows."""
    window_data_path = Path(window_data_path)
    metadata_path = Path(metadata_path)
    predictions_output_path = Path(predictions_output_path)
    summary_output_path = Path(summary_output_path)

    if not window_data_path.exists():
        raise FileNotFoundError(
            f"IDS window data not found: {window_data_path}. "
            "Run ids-run-inference-step3 first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"IDS window metadata not found: {metadata_path}. "
            "Run ids-run-inference-step3 first."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")

    device = _resolve_ids_prediction_device(device_setting)
    windows = np.load(window_data_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    label_names = _load_lite_v6_label_names()
    model, model_config = _load_lite_v6_trained_model(device)

    _validate_ids_prediction_inputs(windows, metadata, model_config, label_names)

    evaluator = Evaluator(
        model=model,
        device=device,
        output_dir=str(IDS_INFERENCE_OUTPUT_DIR),
    )

    prediction_tables = []
    if "source_file_name" not in metadata.columns:
        metadata["source_file_name"] = "unknown_file"

    for source_file_name, group in metadata.groupby("source_file_name", sort=False):
        group = group.sort_values("window_index").copy()
        group_indices = group.index.to_numpy(dtype=int)
        group_windows = np.asarray(windows[group_indices], dtype=np.float32)

        print(
            "[ids-inference-step4] Predicting "
            f"{len(group_windows)} windows from {source_file_name}"
        )
        predictions, probabilities = _predict_ids_window_group(
            evaluator=evaluator,
            windows=group_windows,
            batch_size=batch_size,
            device=device,
        )
        prediction_tables.append(
            _build_ids_prediction_rows(
                group_metadata=group,
                predictions=predictions,
                probabilities=probabilities,
                label_names=label_names,
            )
        )

    if prediction_tables:
        predictions_df = pd.concat(prediction_tables, ignore_index=True)
    else:
        predictions_df = pd.DataFrame()

    evaluator.save_ids_predictions(predictions_df, predictions_output_path)

    prediction_counts = {}
    if "predicted_label_name" in predictions_df.columns:
        prediction_counts = (
            predictions_df["predicted_label_name"].value_counts().sort_index().to_dict()
        )

    expected_counts = {}
    expected_accuracy = None
    if "expected_label_name" in predictions_df.columns:
        expected_counts = (
            predictions_df["expected_label_name"].value_counts().sort_index().to_dict()
        )
    if "expected_match" in predictions_df.columns and len(predictions_df) > 0:
        expected_accuracy = float(predictions_df["expected_match"].mean())

    summary = {
        "window_data_path": str(window_data_path),
        "metadata_path": str(metadata_path),
        "predictions_output_path": str(predictions_output_path),
        "summary_output_path": str(summary_output_path),
        "model_path": str(LITE_V6_MODEL_PATH),
        "model_config_path": str(LITE_V6_MODEL_CONFIG_PATH),
        "label_classes_path": str(LITE_V6_LABEL_CLASSES_PATH),
        "device": device,
        "batch_size": batch_size,
        "window_shape": tuple(windows.shape),
        "prediction_rows": int(len(predictions_df)),
        "source_files": int(metadata["source_file_name"].nunique()),
        "prediction_counts": prediction_counts,
        "expected_counts": expected_counts,
        "expected_accuracy": expected_accuracy,
    }

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output_path, "wb") as f:
        pickle.dump(summary, f)

    return {
        "predictions": predictions_df,
        "summary": summary,
    }


def run_step_4(
    window_data_path: Path = STEP_3_WINDOW_DATA_PATH,
    metadata_path: Path = STEP_3_WINDOW_METADATA_PATH,
    predictions_output_path: Path = IDS_PREDICTIONS_PATH,
    summary_output_path: Path = IDS_PREDICTION_SUMMARY_PATH,
    batch_size: int = IDS_PREDICTION_BATCH_SIZE,
    device_setting: str = IDS_PREDICTION_DEVICE,
) -> dict:
    """AWS IDS inference STEP=4: predict window labels with Lite V6."""
    return predict_ids_windows(
        window_data_path=window_data_path,
        metadata_path=metadata_path,
        predictions_output_path=predictions_output_path,
        summary_output_path=summary_output_path,
        batch_size=batch_size,
        device_setting=device_setting,
    )


def _path_report(path: Path) -> dict:
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def _value_counts_report(series: pd.Series) -> dict:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def debug_ids_inference_artifacts(
    step_2_data_path: Path = STEP_2_PROCESSED_DATA_PATH,
    window_data_path: Path = STEP_3_WINDOW_DATA_PATH,
    metadata_path: Path = STEP_3_WINDOW_METADATA_PATH,
    features_used_path: Path = STEP_3_FEATURES_USED_PATH,
) -> dict:
    """Read-only debug report for AWS IDS Lite V6 inference artifacts."""
    model_config = _load_json(LITE_V6_MODEL_CONFIG_PATH)
    label_names = _load_lite_v6_label_names()
    selected_features = list(_load_pickle(LITE_V6_SELECTED_FEATURES_PATH))
    scaler = _load_pickle(LITE_V6_STANDARD_SCALER_PATH)
    window_config = _load_pickle(LITE_V6_WINDOW_CONFIG_PATH)

    used_features = []
    if Path(features_used_path).exists():
        used_features = list(_load_pickle(Path(features_used_path)))

    df = pd.read_pickle(step_2_data_path) if Path(step_2_data_path).exists() else None
    windows = np.load(window_data_path, mmap_mode="r") if Path(window_data_path).exists() else None
    metadata = pd.read_csv(metadata_path) if Path(metadata_path).exists() else None

    missing_selected_features = []
    label_counts = {}
    source_file_counts = {}
    selected_nan_count = None
    selected_inf_count = None
    constant_feature_count = None
    constant_feature_preview = []

    if df is not None:
        missing_selected_features = [
            feature for feature in selected_features
            if feature not in df.columns
        ]
        if "expected_label_name" in df.columns:
            label_counts = _value_counts_report(df["expected_label_name"])
        if "source_file_name" in df.columns:
            source_file_counts = _value_counts_report(df["source_file_name"])

        if not missing_selected_features:
            feature_df = df[selected_features].apply(pd.to_numeric, errors="coerce")
            selected_nan_count = int(feature_df.isna().sum().sum())
            selected_inf_count = int(
                np.isinf(feature_df.to_numpy(dtype=np.float64, copy=False)).sum()
            )
            constant_features = [
                feature
                for feature in selected_features
                if int(feature_df[feature].nunique(dropna=False)) <= 1
            ]
            constant_feature_count = len(constant_features)
            constant_feature_preview = constant_features[:25]

    scaler_feature_count = getattr(scaler, "n_features_in_", None)
    if scaler_feature_count is None and hasattr(scaler, "mean_"):
        scaler_feature_count = len(scaler.mean_)
    scaler_feature_count = (
        int(scaler_feature_count) if scaler_feature_count is not None else None
    )

    scaler_feature_names = list(getattr(scaler, "feature_names_in_", []))
    scaler_feature_names_match = (
        scaler_feature_names == selected_features if scaler_feature_names else None
    )

    window_shape = list(windows.shape) if windows is not None else None
    metadata_rows = int(len(metadata)) if metadata is not None else None
    metadata_counts = {}
    if metadata is not None and "source_file_name" in metadata.columns:
        metadata_counts = _value_counts_report(metadata["source_file_name"])

    contract = {
        "selected_features_match_step3_used_features": selected_features == used_features,
        "scaler_feature_count_matches_selected_features": (
            scaler_feature_count == len(selected_features)
        ),
        "label_count_matches_model_c_out": len(label_names) == int(model_config["c_out"]),
        "window_shape_matches_model": (
            windows is not None
            and windows.ndim == 3
            and int(windows.shape[1]) == int(model_config["seq_len"])
            and int(windows.shape[2]) == int(model_config["c_in"])
        ),
        "metadata_rows_match_window_count": (
            windows is not None
            and metadata is not None
            and int(len(metadata)) == int(len(windows))
        ),
        "step2_has_all_selected_features": not missing_selected_features,
    }

    return {
        "paths": {
            "step_2_data": _path_report(step_2_data_path),
            "window_data": _path_report(window_data_path),
            "window_metadata": _path_report(metadata_path),
            "step_3_features_used": _path_report(features_used_path),
            "lite_v6_model": _path_report(LITE_V6_MODEL_PATH),
            "lite_v6_model_config": _path_report(LITE_V6_MODEL_CONFIG_PATH),
            "lite_v6_selected_features": _path_report(LITE_V6_SELECTED_FEATURES_PATH),
            "lite_v6_standard_scaler": _path_report(LITE_V6_STANDARD_SCALER_PATH),
            "lite_v6_window_config": _path_report(LITE_V6_WINDOW_CONFIG_PATH),
            "lite_v6_label_classes": _path_report(LITE_V6_LABEL_CLASSES_PATH),
        },
        "model_config": {
            "c_in": int(model_config["c_in"]),
            "c_out": int(model_config["c_out"]),
            "seq_len": int(model_config["seq_len"]),
            "model_name": model_config.get("model_name"),
        },
        "label_classes": {
            "count": len(label_names),
            "index_to_label": {index: label for index, label in enumerate(label_names)},
        },
        "features": {
            "selected_feature_count": len(selected_features),
            "step3_used_feature_count": len(used_features),
            "missing_selected_features_in_step2": missing_selected_features,
            "selected_nan_count_before_step3_fill": selected_nan_count,
            "selected_inf_count_before_step3_fill": selected_inf_count,
            "constant_selected_feature_count": constant_feature_count,
            "constant_selected_features_preview": constant_feature_preview,
        },
        "scaler": {
            "type": type(scaler).__name__,
            "feature_count": scaler_feature_count,
            "has_feature_names_in": bool(scaler_feature_names),
            "feature_names_match_selected_features": scaler_feature_names_match,
        },
        "window_config": {
            "best_window_size": int(window_config["best_window_size"]),
            "best_step_size": int(window_config["best_step_size"]),
        },
        "step2_data": {
            "shape": list(df.shape) if df is not None else None,
            "expected_label_counts": label_counts,
            "source_file_row_counts": source_file_counts,
        },
        "step3_windows": {
            "shape": window_shape,
            "metadata_rows": metadata_rows,
            "metadata_windows_by_file": metadata_counts,
        },
        "contract": contract,
    }


def _series_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _feature_stats_table(
    feature_df: pd.DataFrame,
    scaled_features: np.ndarray,
    selected_features: list[str],
    scaler,
) -> pd.DataFrame:
    rows = []
    scaler_means = getattr(scaler, "mean_", np.full(len(selected_features), np.nan))
    scaler_scales = getattr(scaler, "scale_", np.full(len(selected_features), np.nan))

    for feature_index, feature_name in enumerate(selected_features):
        raw_series = feature_df[feature_name]
        scaled_series = scaled_features[:, feature_index]

        rows.append(
            {
                "feature": feature_name,
                "raw_min": _series_float(raw_series.min()),
                "raw_max": _series_float(raw_series.max()),
                "raw_mean": _series_float(raw_series.mean()),
                "raw_std": _series_float(raw_series.std()),
                "raw_zero_fraction": float((raw_series == 0).mean()),
                "raw_unique_count": int(raw_series.nunique(dropna=False)),
                "raw_constant": int(raw_series.nunique(dropna=False)) <= 1,
                "scaler_mean": float(scaler_means[feature_index]),
                "scaler_scale": float(scaler_scales[feature_index]),
                "scaled_min": float(np.nanmin(scaled_series)),
                "scaled_max": float(np.nanmax(scaled_series)),
                "scaled_mean": float(np.nanmean(scaled_series)),
                "scaled_std": float(np.nanstd(scaled_series)),
                "scaled_abs_max": float(np.nanmax(np.abs(scaled_series))),
                "scaled_abs_mean": float(np.nanmean(np.abs(scaled_series))),
                "scaled_abs_gt_3_fraction": float((np.abs(scaled_series) > 3).mean()),
                "scaled_abs_gt_10_fraction": float((np.abs(scaled_series) > 10).mean()),
            }
        )

    return pd.DataFrame(rows)


def _feature_stats_preview(records: pd.DataFrame, columns: list[str]) -> list[dict]:
    preview = records[columns].copy()
    return preview.to_dict(orient="records")


def debug_ids_feature_distribution(
    step_2_data_path: Path = STEP_2_PROCESSED_DATA_PATH,
    top_n: int = 20,
) -> dict:
    """Read-only debug report for AWS IDS feature distribution."""
    step_2_data_path = Path(step_2_data_path)
    if not step_2_data_path.exists():
        raise FileNotFoundError(
            f"STEP=2 output not found: {step_2_data_path}. "
            "Run ids-run-inference-step2 first."
        )

    selected_features = list(_load_pickle(LITE_V6_SELECTED_FEATURES_PATH))
    scaler = _load_pickle(LITE_V6_STANDARD_SCALER_PATH)
    df = pd.read_pickle(step_2_data_path)

    missing_features = [
        feature for feature in selected_features
        if feature not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "STEP=2 data is missing Lite V6 selected features: "
            f"{missing_features}"
        )

    raw_feature_df = df[selected_features].apply(pd.to_numeric, errors="coerce")
    clean_feature_df = raw_feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled_features = scaler.transform(clean_feature_df)

    stats = _feature_stats_table(
        feature_df=clean_feature_df,
        scaled_features=scaled_features,
        selected_features=selected_features,
        scaler=scaler,
    )

    constant_features = stats.loc[stats["raw_constant"], "feature"].tolist()
    almost_all_zero = stats.loc[
        stats["raw_zero_fraction"] >= 0.99, "feature"
    ].tolist()

    top_scaled_abs_max = stats.sort_values(
        "scaled_abs_max", ascending=False
    ).head(top_n)
    top_scaled_abs_mean = stats.sort_values(
        "scaled_abs_mean", ascending=False
    ).head(top_n)
    top_zero_fraction = stats.sort_values(
        "raw_zero_fraction", ascending=False
    ).head(top_n)

    per_label_report = {}
    if "expected_label_name" in df.columns:
        for label_name, label_group in df.groupby("expected_label_name", sort=True):
            label_features = (
                label_group[selected_features]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
            per_label_report[str(label_name)] = {
                "rows": int(len(label_group)),
                "constant_feature_count": int(
                    sum(
                        int(label_features[feature].nunique(dropna=False)) <= 1
                        for feature in selected_features
                    )
                ),
                "zero_fraction_mean": float((label_features == 0).mean().mean()),
            }

    return {
        "step_2_data_path": str(step_2_data_path),
        "rows": int(len(df)),
        "selected_feature_count": int(len(selected_features)),
        "constant_feature_count": int(len(constant_features)),
        "constant_features": constant_features,
        "almost_all_zero_feature_count": int(len(almost_all_zero)),
        "almost_all_zero_features": almost_all_zero,
        "raw_nan_count_before_fill": int(raw_feature_df.isna().sum().sum()),
        "raw_inf_count_before_fill": int(
            np.isinf(raw_feature_df.to_numpy(dtype=np.float64, copy=False)).sum()
        ),
        "top_scaled_abs_max": _feature_stats_preview(
            top_scaled_abs_max,
            [
                "feature",
                "raw_min",
                "raw_max",
                "raw_mean",
                "scaler_mean",
                "scaler_scale",
                "scaled_abs_max",
                "scaled_abs_gt_10_fraction",
            ],
        ),
        "top_scaled_abs_mean": _feature_stats_preview(
            top_scaled_abs_mean,
            [
                "feature",
                "raw_mean",
                "raw_std",
                "scaler_mean",
                "scaler_scale",
                "scaled_abs_mean",
                "scaled_abs_gt_3_fraction",
            ],
        ),
        "top_zero_fraction": _feature_stats_preview(
            top_zero_fraction,
            [
                "feature",
                "raw_zero_fraction",
                "raw_unique_count",
                "raw_constant",
                "raw_min",
                "raw_max",
            ],
        ),
        "per_expected_label": per_label_report,
    }


def debug_ids_selected_feature_values(
    step_2_data_path: Path = STEP_2_PROCESSED_DATA_PATH,
    features: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Read-only report for selected raw and scaled AWS IDS feature values."""
    step_2_data_path = Path(step_2_data_path)
    if not step_2_data_path.exists():
        raise FileNotFoundError(
            f"STEP=2 output not found: {step_2_data_path}. "
            "Run ids-run-inference-step2 first."
        )

    if not features:
        raise ValueError("At least one feature must be provided with --features.")

    selected_features = list(_load_pickle(LITE_V6_SELECTED_FEATURES_PATH))
    scaler = _load_pickle(LITE_V6_STANDARD_SCALER_PATH)
    df = pd.read_pickle(step_2_data_path)

    missing_from_step2 = [feature for feature in features if feature not in df.columns]
    missing_from_selected = [
        feature for feature in features if feature not in selected_features
    ]

    available_features = [
        feature
        for feature in features
        if feature in df.columns and feature in selected_features
    ]

    feature_to_selected_index = {
        feature: index for index, feature in enumerate(selected_features)
    }

    report_by_label = {}
    label_column = "expected_label_name" if "expected_label_name" in df.columns else "Label"

    for label_name, label_group in df.groupby(label_column, sort=True):
        label_rows = {"rows": int(len(label_group)), "features": {}}

        for feature in available_features:
            feature_index = feature_to_selected_index[feature]
            raw_series = pd.to_numeric(label_group[feature], errors="coerce")
            clean_series = raw_series.replace([np.inf, -np.inf], np.nan).fillna(0)

            scaler_mean = float(scaler.mean_[feature_index])
            scaler_scale = float(scaler.scale_[feature_index])
            scaled_series = (clean_series - scaler_mean) / scaler_scale

            label_rows["features"][feature] = {
                "raw_min": _series_float(clean_series.min()),
                "raw_max": _series_float(clean_series.max()),
                "raw_mean": _series_float(clean_series.mean()),
                "raw_std": _series_float(clean_series.std()),
                "raw_zero_fraction": float((clean_series == 0).mean()),
                "raw_unique_count": int(clean_series.nunique(dropna=False)),
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
                "scaled_min": _series_float(scaled_series.min()),
                "scaled_max": _series_float(scaled_series.max()),
                "scaled_mean": _series_float(scaled_series.mean()),
                "scaled_std": _series_float(scaled_series.std()),
            }

        report_by_label[str(label_name)] = label_rows

    return {
        "step_2_data_path": str(step_2_data_path),
        "rows": int(len(df)),
        "label_column": label_column,
        "requested_features": list(features),
        "available_features": available_features,
        "missing_from_step2": missing_from_step2,
        "missing_from_lite_v6_selected_features": missing_from_selected,
        "per_label": report_by_label,
    }


def debug_ids_prediction_probabilities(
    predictions_path: Path = IDS_PREDICTIONS_PATH,
    top_k: int = 5,
    sample_windows_per_file: int = 5,
) -> dict:
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_path}. "
            "Run ids-run-inference-step4 first."
        )

    predictions = pd.read_csv(predictions_path)
    label_names = _load_lite_v6_label_names()

    probability_columns = []
    for class_index, label_name in enumerate(label_names):
        column_name = f"prob_{class_index}_{_safe_probability_column_name(label_name)}"
        if column_name in predictions.columns:
            probability_columns.append((class_index, label_name, column_name))

    if not probability_columns:
        raise ValueError(f"No probability columns found in: {predictions_path}")

    report_by_file = {}
    for source_file_name, group in predictions.groupby("source_file_name", sort=False):
        group = group.sort_values("window_index").copy()
        probability_matrix = group[
            [column for _, _, column in probability_columns]
        ].to_numpy()

        average_probabilities = probability_matrix.mean(axis=0)
        average_top_indices = np.argsort(average_probabilities)[::-1][:top_k]
        average_top_classes = [
            {
                "class_index": int(probability_columns[index][0]),
                "class_name": probability_columns[index][1],
                "average_probability": float(average_probabilities[index]),
            }
            for index in average_top_indices
        ]

        sample_windows = []
        sample_rows = group.head(sample_windows_per_file).reset_index(drop=True)
        for row_position, row in sample_rows.iterrows():
            row_probabilities = probability_matrix[row_position]
            top_indices = np.argsort(row_probabilities)[::-1][:top_k]
            sample_windows.append(
                {
                    "window_index": int(row["window_index"]),
                    "flow_range": row.get("flow_range"),
                    "expected_label_name": row.get("expected_label_name"),
                    "predicted_label_name": row.get("predicted_label_name"),
                    "confidence": float(row.get("confidence")),
                    "top_classes": [
                        {
                            "class_index": int(probability_columns[index][0]),
                            "class_name": probability_columns[index][1],
                            "probability": float(row_probabilities[index]),
                        }
                        for index in top_indices
                    ],
                }
            )

        report_by_file[str(source_file_name)] = {
            "windows": int(len(group)),
            "prediction_counts": _value_counts_report(group["predicted_label_name"]),
            "confidence_min": float(group["confidence"].min()),
            "confidence_max": float(group["confidence"].max()),
            "confidence_mean": float(group["confidence"].mean()),
            "average_top_classes": average_top_classes,
            "sample_windows": sample_windows,
        }

    return {
        "predictions_path": str(predictions_path),
        "rows": int(len(predictions)),
        "top_k": int(top_k),
        "sample_windows_per_file": int(sample_windows_per_file),
        "files": report_by_file,
    }


def _window_feature_profile(windows: np.ndarray) -> dict:
    """Summarize model-ready windows across all flows and selected features."""
    windows = np.asarray(windows, dtype=np.float32)
    flattened = windows.reshape(-1, windows.shape[-1])

    return {
        "window_count": int(windows.shape[0]),
        "flow_rows": int(flattened.shape[0]),
        "feature_mean": np.mean(flattened, axis=0),
        "feature_std": np.std(flattened, axis=0),
        "feature_min": np.min(flattened, axis=0),
        "feature_max": np.max(flattened, axis=0),
        "feature_zero_fraction": np.mean(flattened == 0, axis=0),
    }


def _safe_float(value) -> float | None:
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def _profile_distance(aws_profile: dict, lite_profile: dict) -> dict:
    """Measure how far an AWS IDS profile is from one Lite V6 class profile."""
    lite_std = lite_profile["feature_std"].astype(np.float64)
    valid_scale = lite_std > 1e-12

    deltas = np.full_like(lite_std, np.nan, dtype=np.float64)
    deltas[valid_scale] = np.abs(
        (
            aws_profile["feature_mean"][valid_scale]
            - lite_profile["feature_mean"][valid_scale]
        )
        / lite_std[valid_scale]
    )

    finite = np.isfinite(deltas)
    if not finite.any():
        return {
            "mean_abs_standardized_delta": None,
            "median_abs_standardized_delta": None,
            "feature_count_used": 0,
        }

    return {
        "mean_abs_standardized_delta": float(np.mean(deltas[finite])),
        "median_abs_standardized_delta": float(np.median(deltas[finite])),
        "feature_count_used": int(finite.sum()),
    }


def _top_profile_differences(
    aws_profile: dict,
    lite_profile: dict,
    selected_features: list[str],
    top_n: int,
) -> list[dict]:
    """Show which selected features differ most from the matching Lite V6 class."""
    lite_std = lite_profile["feature_std"].astype(np.float64)
    valid_scale = lite_std > 1e-12

    deltas = np.full_like(lite_std, np.nan, dtype=np.float64)
    deltas[valid_scale] = np.abs(
        (
            aws_profile["feature_mean"][valid_scale]
            - lite_profile["feature_mean"][valid_scale]
        )
        / lite_std[valid_scale]
    )

    ordered_indices = np.argsort(np.nan_to_num(deltas, nan=-1.0))[::-1]
    rows = []
    for feature_index in ordered_indices[:top_n]:
        if not np.isfinite(deltas[feature_index]):
            continue

        rows.append(
            {
                "feature": selected_features[feature_index],
                "abs_standardized_delta": float(deltas[feature_index]),
                "aws_mean": _safe_float(aws_profile["feature_mean"][feature_index]),
                "lite_mean": _safe_float(lite_profile["feature_mean"][feature_index]),
                "lite_std": _safe_float(lite_profile["feature_std"][feature_index]),
                "aws_zero_fraction": _safe_float(
                    aws_profile["feature_zero_fraction"][feature_index]
                ),
                "lite_zero_fraction": _safe_float(
                    lite_profile["feature_zero_fraction"][feature_index]
                ),
            }
        )

    return rows


def _load_lite_v6_window_pools() -> dict:
    """Load Lite V6 model-ready train/val/test windows with memory mapping."""
    return {
        "train": (
            np.load(LITE_V6_X_TRAIN_PATH, mmap_mode="r"),
            np.load(LITE_V6_Y_TRAIN_PATH, mmap_mode="r"),
        ),
        "val": (
            np.load(LITE_V6_X_VAL_PATH, mmap_mode="r"),
            np.load(LITE_V6_Y_VAL_PATH, mmap_mode="r"),
        ),
        "test": (
            np.load(LITE_V6_X_TEST_PATH, mmap_mode="r"),
            np.load(LITE_V6_Y_TEST_PATH, mmap_mode="r"),
        ),
    }


def _sample_lite_windows_for_label(
    lite_pools: dict,
    label_index: int,
    sample_limit: int,
) -> tuple[np.ndarray | None, dict]:
    """Sample Lite V6 windows for one class without loading all windows."""
    split_matches = {}
    split_counts = {}
    total_matches = 0

    for split_name, (_, y_values) in lite_pools.items():
        y_array = np.asarray(y_values)
        if y_array.ndim > 1 and y_array.shape[-1] > 1:
            y_array = np.argmax(y_array, axis=1)
        else:
            y_array = y_array.reshape(-1)

        matches = np.flatnonzero(y_array.astype(int) == int(label_index))
        split_matches[split_name] = matches
        split_counts[split_name] = int(len(matches))
        total_matches += int(len(matches))

    if total_matches == 0:
        return None, {
            "available": 0,
            "sampled": 0,
            "available_by_split": split_counts,
            "sampled_by_split": {},
        }

    if sample_limit and total_matches > sample_limit:
        global_positions = np.linspace(
            0,
            total_matches - 1,
            sample_limit,
            dtype=np.int64,
        )
    else:
        global_positions = np.arange(total_matches, dtype=np.int64)

    chunks = []
    sampled_by_split = {}
    offset = 0
    for split_name, (x_values, _) in lite_pools.items():
        matches = split_matches[split_name]
        split_count = len(matches)
        in_split = global_positions[
            (global_positions >= offset)
            & (global_positions < offset + split_count)
        ] - offset
        selected_indices = matches[in_split]
        sampled_by_split[split_name] = int(len(selected_indices))

        if len(selected_indices) > 0:
            chunks.append(np.asarray(x_values[selected_indices], dtype=np.float32))

        offset += split_count

    return np.concatenate(chunks, axis=0), {
        "available": int(total_matches),
        "sampled": int(sum(sampled_by_split.values())),
        "available_by_split": split_counts,
        "sampled_by_split": sampled_by_split,
    }


def debug_ids_compare_lite_v6_classes(
    window_data_path: Path = STEP_3_WINDOW_DATA_PATH,
    metadata_path: Path = STEP_3_WINDOW_METADATA_PATH,
    labels: tuple[str, ...] | list[str] | None = None,
    sample_windows_per_label: int | None = None,
    top_n_features: int = 15,
) -> dict:
    """Compare AWS IDS model-ready windows against Lite V6 class windows."""
    window_data_path = Path(window_data_path)
    metadata_path = Path(metadata_path)
    labels_to_compare = list(labels or IDS_DEBUG_COMPARE_LABELS)
    sample_limit = (
        IDS_DEBUG_COMPARE_SAMPLE_WINDOWS_PER_LABEL
        if sample_windows_per_label is None
        else int(sample_windows_per_label)
    )

    if not window_data_path.exists():
        raise FileNotFoundError(
            f"IDS window data not found: {window_data_path}. "
            "Run ids-run-inference-step3 first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"IDS window metadata not found: {metadata_path}. "
            "Run ids-run-inference-step3 first."
        )

    selected_features = list(_load_pickle(LITE_V6_SELECTED_FEATURES_PATH))
    label_names = _load_lite_v6_label_names()
    label_to_index = {label_name: index for index, label_name in enumerate(label_names)}

    aws_windows = np.load(window_data_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    if "expected_label_name" not in metadata.columns:
        raise ValueError("Window metadata must contain expected_label_name.")

    lite_pools = _load_lite_v6_window_pools()

    aws_profiles = {}
    aws_window_groups = {}
    for label_name, group in metadata.groupby("expected_label_name", sort=True):
        if str(label_name) not in labels_to_compare:
            continue

        group = group.sort_values("window_index")
        group_indices = group.index.to_numpy(dtype=int)
        group_windows = np.asarray(aws_windows[group_indices], dtype=np.float32)
        aws_profiles[str(label_name)] = _window_feature_profile(group_windows)
        aws_window_groups[str(label_name)] = int(len(group_windows))

    lite_profiles = {}
    lite_samples = {}
    missing_lite_labels = []
    for label_name in labels_to_compare:
        if label_name not in label_to_index:
            missing_lite_labels.append(label_name)
            continue

        label_windows, sample_report = _sample_lite_windows_for_label(
            lite_pools=lite_pools,
            label_index=label_to_index[label_name],
            sample_limit=sample_limit,
        )
        lite_samples[label_name] = sample_report

        if label_windows is not None:
            lite_profiles[label_name] = _window_feature_profile(label_windows)

    closest_lite_classes_by_aws_label = {}
    top_differences_against_same_label = {}
    for aws_label, aws_profile in aws_profiles.items():
        distances = []
        for lite_label, lite_profile in lite_profiles.items():
            distance = _profile_distance(aws_profile, lite_profile)
            distances.append(
                {
                    "lite_label": lite_label,
                    "lite_label_index": int(label_to_index[lite_label]),
                    "lite_windows_sampled": int(
                        lite_samples[lite_label]["sampled"]
                    ),
                    **distance,
                }
            )

        distances.sort(
            key=lambda row: (
                float("inf")
                if row["mean_abs_standardized_delta"] is None
                else row["mean_abs_standardized_delta"]
            )
        )
        closest_lite_classes_by_aws_label[aws_label] = distances

        if aws_label in lite_profiles:
            top_differences_against_same_label[aws_label] = _top_profile_differences(
                aws_profile=aws_profile,
                lite_profile=lite_profiles[aws_label],
                selected_features=selected_features,
                top_n=top_n_features,
            )

    return {
        "paths": {
            "aws_window_data": _path_report(window_data_path),
            "aws_window_metadata": _path_report(metadata_path),
            "lite_v6_x_train": _path_report(LITE_V6_X_TRAIN_PATH),
            "lite_v6_x_val": _path_report(LITE_V6_X_VAL_PATH),
            "lite_v6_x_test": _path_report(LITE_V6_X_TEST_PATH),
            "lite_v6_y_train": _path_report(LITE_V6_Y_TRAIN_PATH),
            "lite_v6_y_val": _path_report(LITE_V6_Y_VAL_PATH),
            "lite_v6_y_test": _path_report(LITE_V6_Y_TEST_PATH),
        },
        "comparison_basis": (
            "model-ready windows after selected features, Lite V6 scaler, "
            "window_size=30, step_size=5"
        ),
        "interpretation": (
            "Lower mean_abs_standardized_delta means the AWS IDS windows are "
            "closer to that Lite V6 class distribution."
        ),
        "labels_compared": labels_to_compare,
        "sample_windows_per_lite_label": int(sample_limit),
        "selected_feature_count": int(len(selected_features)),
        "aws_window_groups": aws_window_groups,
        "lite_window_samples": lite_samples,
        "missing_lite_labels": missing_lite_labels,
        "closest_lite_classes_by_aws_label": closest_lite_classes_by_aws_label,
        "top_differences_against_same_label": top_differences_against_same_label,
    }

if __name__ == "__main__":
    import logging
    import numpy as np
    import torch
    from pathlib import Path

    # ==================== IMPORT CONFIG ====================
    from config.config import (
        BASE_DIR, MODELS_DIR, RESULTS_DIR, EVAL_BATCH_SIZE, USE_GPU,
        LITE_USE_SHORT_CONV_BRANCH, LITE_CONV_KERNEL_SIZE, LITE_CONV_NUM_LAYERS,
        LITE_FUSION_TYPE, LITE_ATTENTION_FRACTION
    )
    from models.model_config import create_model

    # ==================== LOGGING ====================
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("\n" + "█" * 80)
    logger.info("█" + " " * 25 + "STEP 8: STANDALONE EVALUATION" + " " * 25 + "█")
    logger.info("█" * 80)

    # ==================== DEVICE ====================
    device = resolve_device(USE_GPU)
    logger.info(f"\n🖥️ Using device: {device}")

    # ==================== LOAD TEST DATA ====================
    DATA_PATH = BASE_DIR / "outputs" / "data"

    X_test = np.load(DATA_PATH / "X_test.npy", mmap_mode="r")
    y_test = np.load(DATA_PATH / "y_test.npy")

    logger.info(f"\n📂 Loaded test data:")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_test: {y_test.shape}")

    # ==================== LOAD LABEL ENCODER ====================
    import pickle

    le_path = Path(MODELS_DIR) / "label_encoder.pkl"
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # Validate test labels against saved encoder
    unique_test_labels = np.unique(y_test)
    encoder_classes = set(le.classes_)
    unknown_labels = set(unique_test_labels) - encoder_classes
    if unknown_labels:
        raise ValueError(
            f"Test labels contain unseen classes relative to label encoder: {sorted(unknown_labels)}"
        )

    logger.info(f"Label encoder classes: {le.classes_.tolist()}")
    logger.info(f"Unique raw test labels before transform: {unique_test_labels.tolist()}")

    # 🔥 APPLY TRANSFORMATION HERE
    y_test = le.transform(y_test)

    logger.info(f"✓ Label encoder loaded and applied to test labels")
    logger.info(f"Unique test labels after transform: {np.unique(y_test)}")

    # ==================== LOAD MODEL CONFIG ====================
    import json

    config_path = Path(MODELS_DIR) / "model_config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    logger.info(f"\n✓ Loaded model config")

    # ==================== REBUILD MODEL ====================
    model, _ = create_model(
        c_in=config_dict["c_in"],
        c_out=config_dict["c_out"],
        seq_len=config_dict["seq_len"],
        d_model=config_dict["d_model"],
        n_heads=config_dict["n_heads"],
        d_ff=config_dict["d_ff"],
        dropout=config_dict["dropout"],
        activation=config_dict["activation"],
        n_layers=config_dict["n_layers"],
        fc_dropout=config_dict["fc_dropout"],
        use_cnn_local_branch=config_dict.get("use_cnn_local_branch", LITE_USE_SHORT_CONV_BRANCH),
        conv_kernel_size=config_dict.get("conv_kernel_size", LITE_CONV_KERNEL_SIZE),
        cnn_num_conv_layers=config_dict.get("cnn_num_conv_layers", LITE_CONV_NUM_LAYERS),
        fusion_type=config_dict.get("fusion_type", LITE_FUSION_TYPE),
        attention_fraction=config_dict.get("attention_fraction", LITE_ATTENTION_FRACTION),
        device=device
    )

    # ==================== LOAD TRAINED WEIGHTS ====================
    best_model_path = Path(MODELS_DIR) / "best_model.pt"
    final_model_path = Path(MODELS_DIR) / "final_model.pt"
    model_path = best_model_path if best_model_path.exists() else final_model_path
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture does not match the current Lite_V6 model. "
            "Train Lite_V6 first to create a compatible checkpoint."
        ) from exc

    logger.info(f"✓ Loaded trained model from {model_path}")

    # ==================== CREATE TEST LOADER ====================
    test_dataset = NumpyWindowDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=(device == "cuda")
    )

    # ==================== EVALUATION ====================
    evaluator = Evaluator(
        model=model,
        device=device,
        output_dir=str(RESULTS_DIR)
    )

    criterion = torch.nn.CrossEntropyLoss()

    results = evaluator.evaluate(test_loader, criterion)

    # ==================== SAVE RESULTS ====================
    evaluator.save_results(results)
    evaluator.save_predictions(results)
    evaluator.plot_confusion_matrix(np.array(results['confusion_matrix']))

    logger.info("\n✅ Standalone evaluation completed!")            
