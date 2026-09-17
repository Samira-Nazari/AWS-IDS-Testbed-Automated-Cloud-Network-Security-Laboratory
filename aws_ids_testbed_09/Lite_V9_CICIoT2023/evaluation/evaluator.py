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
from pathlib import Path
from numpy_window_dataset import NumpyWindowDataset

logger = logging.getLogger(__name__)


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
            "Checkpoint architecture does not match the current Lite_V9 model. "
            "Train Lite_V9 first to create a compatible checkpoint."
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
