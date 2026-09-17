"""
Training module for TST model
Handles training loop with metrics tracking and mid-results display
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
# from pathlib import Path
import time
import logging
from sklearn.metrics import balanced_accuracy_score, f1_score
from numpy_window_dataset import NumpyWindowDataset


logger = logging.getLogger(__name__)


def _labels_from_dataset(dataset) -> np.ndarray:
    """Return labels without forcing large feature arrays into memory."""
    if hasattr(dataset, "y"):
        return np.asarray(dataset.y)
    if hasattr(dataset, "tensors"):
        return dataset.tensors[1].detach().cpu().numpy()
    raise AttributeError("Dataset must expose labels as .y or .tensors[1].")


class WeightedFocalLoss(nn.Module):
    """Standard weighted focal loss for multi-class classification.

    FL = -w_y * (1 - p_y)^gamma * log(p_y)
    The focal factor is computed from the true class probability p_y,
    not from weighted cross entropy.
    """

    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = nn.functional.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        ce_loss = -log_pt
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.weight is not None:
            sample_weights = self.weight.to(logits.device)[targets]
            focal_loss = sample_weights * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class Trainer:
    """Train TST model on CICIoT data"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu',
                 output_dir: str = 'outputs/models'):
        """
        Args:
            model: PyTorch model to train
            device: Device to train on ('cuda' or 'cpu')
            output_dir: Directory to save models
        """
        # self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ✅ FIX: allow model to be None
        self.model = model.to(device) if model is not None else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_macro_f1s = []
        self.val_balanced_accs = []
    
    def create_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          batch_size: int = 64, eval_batch_size: int = 128,
                          use_weighted_sampler: bool = False,
                          sampler_weight_power: float = 0.5) -> tuple:
        """
        Create PyTorch dataloaders
        
        Args:
            X_train: Training features (samples, seq_len, features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            use_weighted_sampler: If True, sample rare classes more often during training
            sampler_weight_power: Strength of inverse-frequency sample weighting
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: CREATING DATA LOADERS")
        logger.info("="*60)
        
        logger.info(f"\nTraining data:")
        logger.info(f"  X shape: {X_train.shape}")
        logger.info(f"  y shape: {y_train.shape}")
        logger.info(f"  Batch size: {batch_size}")
        
        logger.info(f"\nValidation data:")
        logger.info(f"  X shape: {X_val.shape}")
        logger.info(f"  y shape: {y_val.shape}")
        logger.info(f"  Batch size: {eval_batch_size}")
        
        # Keep the huge window arrays lazy; each sample is converted to float32
        # by the dataset and each mini-batch is moved to GPU inside the loop.
        train_dataset = NumpyWindowDataset(X_train, y_train)
        val_dataset = NumpyWindowDataset(X_val, y_val)
        # ✅ ADD HERE
        logger.info(f"Train dataset length: {len(train_dataset)}")
        logger.info(f"Validation dataset length: {len(val_dataset)}")
        
        # Create dataloaders
        if use_weighted_sampler:
            classes, counts = np.unique(y_train, return_counts=True)
            class_weight_values = (counts.max() / counts) ** sampler_weight_power

            class_weights = np.ones(int(classes.max()) + 1, dtype=np.float64)
            class_weights[classes.astype(int)] = class_weight_values

            sample_weights = class_weights[y_train.astype(int)]
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )

            logger.info("  WeightedRandomSampler enabled")
            logger.info(f"  Sampler weight power: {sampler_weight_power}")
            logger.info(f"  Sampler class counts: {dict(zip(classes.astype(int), counts.astype(int)))}")
            logger.info(f"  Sampler class weights: {class_weights}")

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=0,
                pin_memory=(self.device == "cuda"),
            )
        else:
            logger.info("  WeightedRandomSampler disabled")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(self.device == "cuda"),
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device == "cuda"),
        )
        
        logger.info(f"\n✓ DataLoaders created!")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info("="*60 + "\n")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, log_interval: int = 10) -> tuple:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            log_interval: Log metrics every N batches
            
        Returns:
            Tuple of (avg_loss, avg_acc)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(x)
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
            # Log batch progress
            if (batch_idx + 1) % log_interval == 0:
                batch_acc = (preds == y).sum().item() / x.size(0)
                logger.info(f"    Batch [{batch_idx+1:3d}/{len(train_loader)}] "
                           f"Loss: {loss.item():.6f} | Acc: {batch_acc:.4f}")
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> tuple:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (avg_loss, avg_acc, all_preds, all_labels)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                logits = self.model(x)
                loss = criterion(logits, y)
                
                # Accumulate metrics
                total_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
           num_epochs: int = 5, learning_rate: float = 1e-4,
           weight_decay: float = 1e-5, log_interval: int = 10,
           early_stopping_patience: int = 5,
           loss_type: str = "weighted_ce",
           focal_gamma: float = 2.0) -> dict:
        """
        Train model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            log_interval: Log interval
            early_stopping_patience: Epochs to wait for macro-F1 improvement
            loss_type: Loss function type ('weighted_ce' or 'focal')
            focal_gamma: Focusing parameter for focal loss
            
        Returns:
            Dictionary with training history
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 6: TRAINING MODEL")
        logger.info("="*60)
        
        # Setup
        labels = _labels_from_dataset(train_loader.dataset)
        classes, counts = np.unique(labels, return_counts=True)
        weights = np.sqrt(counts.max() / counts)

        class_weights = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
        class_weights[classes.astype(int)] = torch.tensor(weights, dtype=torch.float32)
        class_weights = class_weights.to(self.device)

        logger.info(f"  Class counts: {dict(zip(classes.astype(int), counts.astype(int)))}")
        logger.info(f"  Class weights (sqrt inverse frequency): {class_weights.detach().cpu().numpy()}")

        loss_type = loss_type.lower()
        if loss_type == "focal":
            criterion = WeightedFocalLoss(
                weight=class_weights,
                gamma=focal_gamma,
                reduction="mean",
            )
        elif loss_type == "weighted_ce":
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=learning_rate, weight_decay=weight_decay)
        
        logger.info(f"\nTraining Configuration:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Weight decay: {weight_decay}")
        logger.info(f"  Optimizer: Adam")
        if loss_type == "focal":
            logger.info(f"  Loss function: weighted FocalLoss")
            logger.info(f"  Focal gamma: {focal_gamma}")
        else:
            logger.info(f"  Loss function: weighted CrossEntropyLoss")
        logger.info(f"  Early stopping patience: {early_stopping_patience}")
        logger.info(f"  Device: {self.device}")
        
        best_val_acc = 0.0
        best_val_macro_f1 = -np.inf
        patience_counter = 0
        max_patience = early_stopping_patience
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training
            logger.info(f"\n📚 Training...")
            train_loss, train_acc = self.train_epoch(train_loader, criterion, 
                                                     optimizer, log_interval)
            
            # Validation
            logger.info(f"\n🔍 Validating...")
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
            val_macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            val_weighted_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)

            true_classes, true_counts = np.unique(val_labels, return_counts=True)
            pred_classes, pred_counts = np.unique(val_preds, return_counts=True)
            true_distribution = dict(zip(true_classes.astype(int), true_counts.astype(int)))
            pred_distribution = dict(zip(pred_classes.astype(int), pred_counts.astype(int)))
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.val_macro_f1s.append(val_macro_f1)
            self.val_balanced_accs.append(val_balanced_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch results
            logger.info(f"\n📊 EPOCH {epoch} RESULTS:")
            logger.info(f"  Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f}")
            logger.info(f"  Val Loss:   {val_loss:.6f} | Val Acc:   {val_acc:.6f}")
            logger.info(f"  Val Macro-F1: {val_macro_f1:.6f} | Val Weighted-F1: {val_weighted_f1:.6f}")
            logger.info(f"  Val Balanced Acc: {val_balanced_acc:.6f}")
            logger.info(f"  Val true distribution: {true_distribution}")
            logger.info(f"  Val pred distribution: {pred_distribution}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            
            # Early stopping and checkpointing
            if val_macro_f1 > best_val_macro_f1:
                best_val_acc = val_acc
                best_val_macro_f1 = val_macro_f1
                patience_counter = 0
                logger.info(f"  ✓ New best validation macro-F1! Saving model...")
                self.save_model(f"best_model_epoch_{epoch}.pt")
                self.save_model("best_model.pt")
            else:
                patience_counter += 1
                logger.info(f"  ⚠️  No improvement. Patience: {patience_counter}/{max_patience}")
            
            # Sample predictions
            logger.info(f"\n  Sample predictions (first 10):")
            for i in range(min(10, len(val_preds))):
                logger.info(f"    True: {val_labels[i]} | Pred: {val_preds[i]} | "
                          f"{'✓' if val_labels[i] == val_preds[i] else '✗'}")
            
            logger.info(f"\n")

            if patience_counter >= max_patience:
                logger.info(f"  Early stopping triggered after {patience_counter} epochs without macro-F1 improvement.")
                break
        
        total_time = time.time() - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
        logger.info(f"Final validation accuracy: {self.val_accs[-1]:.6f}")
        logger.info(f"Best validation accuracy: {best_val_acc:.6f}")
        logger.info(f"Best validation macro-F1: {best_val_macro_f1:.6f}")
        logger.info(f"Final training accuracy: {self.train_accs[-1]:.6f}")
        logger.info("="*60 + "\n")
        
        # ✅ Save final model
        self.save_model("final_model.pt")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'val_macro_f1s': self.val_macro_f1s,
            'val_balanced_accs': self.val_balanced_accs,
            'best_val_acc': best_val_acc,
            'best_val_macro_f1': best_val_macro_f1
        }
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        filepath = self.output_dir / filename
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"    Saved: {filepath}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        filepath = self.output_dir / filename
        try:
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture does not match the current Lite_V9 model. "
                "Train Lite_V9 first to create a compatible checkpoint."
            ) from exc
        logger.info(f"✓ Loaded model: {filepath}")

if __name__ == "__main__":
    import numpy as np
    import logging
    from pathlib import Path
    import torch

    

    # ==================== IMPORT CONFIG ====================
    from config.config import (
        BASE_DIR, MODELS_DIR,
        BATCH_SIZE, EVAL_BATCH_SIZE,
        LITE_D_MODEL, LITE_N_HEADS, LITE_D_FF, LITE_DROPOUT,
        LITE_ACTIVATION, LITE_N_LAYERS, LITE_FC_DROPOUT,
        LITE_USE_SHORT_CONV_BRANCH, LITE_CONV_KERNEL_SIZE, LITE_CONV_NUM_LAYERS,
        LITE_FUSION_TYPE, LITE_ATTENTION_FRACTION,
        EPOCHS, EARLY_STOPPING_PATIENCE,
        USE_WEIGHTED_RANDOM_SAMPLER, SAMPLER_WEIGHT_POWER,
        LEARNING_RATE, WEIGHT_DECAY, LOSS_TYPE, FOCAL_GAMMA, LOG_INTERVAL,
        USE_GPU, RANDOM_STATE
    )

    from models.model_config import create_model
    from utils.helpers import set_seed

    # ==================== LOGGING ====================
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    set_seed(RANDOM_STATE)
    logger.info(f"Standalone training seed: {RANDOM_STATE}")

    logger.info("\n" + "█" * 80)
    logger.info("█" + " " * 20 + "TRAINING PIPELINE (STEP 5 → 6 → 7)" + " " * 20 + "█")
    logger.info("█" * 80)

    # ==================== LOAD DATA ====================
    DATA_PATH = BASE_DIR / "outputs" / "data"

    X_train_path = DATA_PATH / "X_train.npy"
    y_train_path = DATA_PATH / "y_train.npy"
    X_val_path   = DATA_PATH / "X_val.npy"
    y_val_path   = DATA_PATH / "y_val.npy"

    if not (X_train_path.exists() and y_train_path.exists() and
            X_val_path.exists() and y_val_path.exists()):
        logger.error("❌ Train/Val data not found!")
        logger.error("Run previous steps first (data + split)")
        exit(1)

    logger.info("\n📂 Loading data...")
    X_train = np.load(X_train_path, mmap_mode="r")
    y_train = np.load(y_train_path)
    X_val   = np.load(X_val_path, mmap_mode="r")
    y_val   = np.load(y_val_path)

    logger.info(f"✓ X_train: {X_train.shape}")
    logger.info(f"✓ y_train: {y_train.shape}")
    logger.info(f"✓ X_val:   {X_val.shape}")
    logger.info(f"✓ y_val:   {y_val.shape}")

    # ✅ ADD HERE
    logger.info(f"y_val size: {len(y_val)}")
    # =================================
    # add this part because I had and error for 
    from sklearn.preprocessing import LabelEncoder
    import pickle   # ✅ add this import

    logger.info("\n🔧 Encoding labels to contiguous values...")
    # le = LabelEncoder()
    # y_train = le.fit_transform(y_train)
    # y_val   = le.transform(y_val)
    le = LabelEncoder()
    le.fit(y_train)

    y_train = le.transform(y_train)
    y_val   = le.transform(y_val)
    logger.info(f"✓ Unique labels after encoding y_train: {np.unique(y_train)}")
    logger.info(f"✓ Unique labels after encoding y_val: {np.unique(y_val)}")

    from collections import Counter

    logger.info(f"Train label distribution: {Counter(y_train)}")
    logger.info(f"Val label distribution: {Counter(y_val)}")

    # ==================== SAVE LABEL ENCODER HERE ====================
    le_path = Path(MODELS_DIR) / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    logger.info(f"✓ Label encoder saved to {le_path}")
    
    # ==================== DEVICE ====================
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
    logger.info(f"\n🖥️ Using device: {device}")

    # ==================== DIMENSIONS ====================
    c_in = X_train.shape[2]
    seq_len = X_train.shape[1]

    # =======================================================
    # Use the train-fitted encoder classes to avoid validation influencing setup.
    n_classes = len(le.classes_)
    # ========================================================
    logger.info("\n📊 Data dimensions:")
    logger.info(f"  c_in: {c_in}")
    logger.info(f"  seq_len: {seq_len}")
    logger.info(f"  n_classes: {n_classes}")

    # ==========================================================
    # ==================== STEP 5 ===============================
    # ==========================================================
    logger.info("\n" + "█" * 80)
    logger.info("█" + " " * 25 + "STEP 5: CREATE DATA LOADERS" + " " * 25 + "█")
    logger.info("█" * 80)

    trainer = Trainer(
        model=None,   # ✅ SAME AS YOUR MAIN
        device=device,
        output_dir=str(MODELS_DIR)
    )

    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        use_weighted_sampler=USE_WEIGHTED_RANDOM_SAMPLER,
        sampler_weight_power=SAMPLER_WEIGHT_POWER
    )

    logger.info("\n🔍 Checking one batch...")
    for x_batch, y_batch in train_loader:
        logger.info(f"Train batch X: {x_batch.shape}")
        logger.info(f"Train batch y: {y_batch.shape}")
        break

    # ==========================================================
    # ==================== STEP 6 ===============================
    # ==========================================================
    logger.info("\n" + "█" * 80)
    logger.info("█" + " " * 25 + "STEP 6: CREATE MODEL" + " " * 29 + "█")
    logger.info("█" * 80)

    model, config = create_model(
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
        device=device
    )

    import json

    config_path = Path(MODELS_DIR) / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info(f"✓ Model config saved to {config_path}")

    # ==========================================================
    # ==================== STEP 7 ===============================
    # ==========================================================
    logger.info("\n" + "█" * 80)
    logger.info("█" + " " * 25 + "STEP 7: TRAIN MODEL" + " " * 30 + "█")
    logger.info("█" * 80)

    trainer = Trainer(   # ✅ RE-CREATE trainer (same as main.py)
        model=model,
        device=device,
        output_dir=str(MODELS_DIR)
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
        focal_gamma=FOCAL_GAMMA
    )

    # ==========================================================
    # ==================== FINAL ================================
    # ==========================================================
    logger.info("\n" + "█" * 80)
    logger.info("█" + " " * 25 + "TRAINING COMPLETED" + " " * 30 + "█")
    logger.info("█" * 80)

    logger.info(f"\nFinal Train Acc: {trainer.train_accs[-1]:.6f}")
    logger.info(f"Final Val Acc:   {trainer.val_accs[-1]:.6f}")

    logger.info("\n✅ STEP 5 + 6 + 7 executed successfully\n")
