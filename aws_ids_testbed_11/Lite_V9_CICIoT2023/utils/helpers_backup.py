"""
Utility helper functions for CICIoT TST project
"""

import numpy as np
import pandas as pd
import torch
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"✓ Random seed set to {seed}")


def setup_logging(log_dir: str = 'outputs/logs', log_file: str = 'training.log'):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        log_file: Log file name
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_filepath = log_path / log_file
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                    val_size: float = 0.2, random_state: int = 42,
                    stratify: bool = True, shuffle: bool = True) -> tuple:
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Feature array
        y: Label array
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining data)
        random_state: Random seed
        stratify: Whether to stratify splits
        shuffle: Whether to shuffle before splitting
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Split into test and temp (train+val)
    test_split = int(n_samples * test_size)
    test_indices = indices[:test_split]
    temp_indices = indices[test_split:]
    
    # Split temp into train and val
    val_split = int(len(temp_indices) * val_size)
    val_indices = temp_indices[:val_split]
    train_indices = temp_indices[val_split:]
    
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    
    logger.info(f"\n✓ Data split completed:")
    logger.info(f"  Train: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def check_data_shapes(X: np.ndarray, y: np.ndarray, expected_dims: int = 3):
    """
    Verify data shapes
    
    Args:
        X: Feature array
        y: Label array
        expected_dims: Expected number of dimensions for X
    """
    logger.info(f"\nData shape validation:")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    
    if len(X.shape) != expected_dims:
        raise ValueError(f"❌ X should be {expected_dims}D, got {len(X.shape)}D")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"❌ X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
    
    logger.info(f"  ✓ Data shapes are valid!")


def get_class_distribution(y: np.ndarray) -> dict:
    """
    Get class distribution statistics
    
    Args:
        y: Label array
        
    Returns:
        Dictionary with class distribution
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = {
        'total_samples': len(y),
        'n_classes': len(unique),
        'class_distribution': {int(cls): int(count) for cls, count in zip(unique, counts)},
        'class_percentages': {int(cls): float(count/len(y)*100) for cls, count in zip(unique, counts)}
    }
    
    logger.info(f"\nClass distribution:")
    logger.info(f"  Total samples: {distribution['total_samples']}")
    logger.info(f"  Number of classes: {distribution['n_classes']}")
    for cls, count in distribution['class_distribution'].items():
        pct = distribution['class_percentages'][cls]
        logger.info(f"    Class {cls}: {count:6d} ({pct:5.2f}%)")
    
    return distribution


def print_config(config: dict):
    """
    Pretty print configuration
    
    Args:
        config: Configuration dictionary
    """
    logger.info("\n" + "="*60)
    logger.info("PROJECT CONFIGURATION")
    logger.info("="*60)
    
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"\n{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("="*60 + "\n")


def get_device(use_gpu: bool = True) -> str:
    """
    Get device (GPU or CPU)
    
    Args:
        use_gpu: Whether to try using GPU
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info(f"✓ Using CPU")
    
    return device


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        filepath: Path to save to
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    logger.info(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load training checkpoint
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load state
        filepath: Path to checkpoint
        device: Device to load to
        
    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    logger.info(f"✓ Checkpoint loaded from {filepath}")
    logger.info(f"  Resuming from epoch {epoch}")
    
    return epoch, metrics

if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    import logging

    # -------------------- CONFIG --------------------
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "outputs" / "data"

    X_path = DATA_PATH / "X_windows.npy"
    y_path = DATA_PATH / "y_windows.npy"

    TEST_SIZE = 0.2
    VALID_SIZE = 0.2
    RANDOM_STATE = 42
    STRATIFY = True
    SHUFFLE = True

    # -------------------- LOGGING --------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("\n" + "█" * 80)
    logger.info("█" + " " * 25 + "STEP 4: DATA SPLITTING (STANDALONE)" + " " * 25 + "█")
    logger.info("█" * 80)

    # -------------------- LOAD DATA --------------------
    if not X_path.exists() or not y_path.exists():
        logger.error("❌ X_windows or y_windows not found!")
        logger.error("Run feature_engineering.py first")
        exit(1)

    logger.info(f"📂 Loading X from: {X_path}")
    logger.info(f"📂 Loading y from: {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    logger.info(f"✓ Loaded X shape: {X.shape}")
    logger.info(f"✓ Loaded y shape: {y.shape}")

    # -------------------- SPLIT DATA --------------------
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        val_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=STRATIFY,
        shuffle=SHUFFLE
    )

    # -------------------- VALIDATE --------------------
    check_data_shapes(X_train, y_train)
    check_data_shapes(X_val, y_val)
    check_data_shapes(X_test, y_test)

    # -------------------- CLASS DISTRIBUTION --------------------
    logger.info("\nTraining set distribution:")
    get_class_distribution(y_train)

    logger.info("\nValidation set distribution:")
    get_class_distribution(y_val)

    logger.info("\nTest set distribution:")
    get_class_distribution(y_test)

    # -------------------- SAVE SPLITS --------------------
    np.save(DATA_PATH / "X_train.npy", X_train)
    np.save(DATA_PATH / "X_val.npy", X_val)
    np.save(DATA_PATH / "X_test.npy", X_test)

    np.save(DATA_PATH / "y_train.npy", y_train)
    np.save(DATA_PATH / "y_val.npy", y_val)
    np.save(DATA_PATH / "y_test.npy", y_test)

    logger.info("\n💾 Saved split datasets:")
    logger.info(f" - X_train.npy, X_val.npy, X_test.npy")
    logger.info(f" - y_train.npy, y_val.npy, y_test.npy")

    logger.info("\n✅ STEP 4 COMPLETE\n")