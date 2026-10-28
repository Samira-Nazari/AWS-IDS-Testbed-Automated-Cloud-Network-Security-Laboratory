import numpy as np
import torch
import logging
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
# from tsai.all import get_splits

import random

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Set random seed for full reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to fix randomness.
    """

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy operations
    np.random.seed(seed)

    # Set seed for PyTorch (CPU)
    torch.manual_seed(seed)

    # If using GPU, set seed for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN (important for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

       # 🔥 ADD IT HERE
    if logger.hasHandlers():
        logger.handlers.clear()
    
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


# =========================================================
# 1) TIME SERIES SPLIT (SKLEARN - CHRONOLOGICAL)
# =========================================================
def split_time_series_sklearn(X, y, n_splits=5):
    """
    Time series split using sklearn TimeSeriesSplit
    (NO SHUFFLE, CHRONOLOGICAL)
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)

    splits = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        logger.info(f"\nFold {fold+1}")
        logger.info(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

        splits.append((train_idx, test_idx))

    return splits


# =========================================================
# 2) TRAIN / VAL / TEST SPLIT (TSAI)
# =========================================================
"""def split_with_tsai(X, y,
                    valid_size=0.2,
                    test_size=0.2,
                    shuffle=False,
                    stratify=True,
                    random_state=42):
    """
    # Split using tsai (recommended for deep learning pipelines)
"""

    splits = get_splits(
        X,
        n_splits=1,
        valid_size=valid_size,
        test_size=test_size,
        shuffle=shuffle,        # ❗ set False for time series
        stratify=stratify,
        random_state=random_state,
        show_plot=True
    )

    train_idx, val_idx, test_idx = splits

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info("\n✓ TSAI Split Completed:")
    logger.info(f"Train: {len(X_train)}")
    logger.info(f"Val:   {len(X_val)}")
    logger.info(f"Test:  {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
"""
#=======================================================
def split_time_series_strict(X, y,
                            valid_size=0.2,
                            test_size=0.2,
                            min_train_ratio=0.5):
    """
    Time-series split with:
    - chronological order preserved
    - all classes guaranteed in train
    - approximate class balance
    
    Args:
        X, y: data
        valid_size, test_size: proportions
        min_train_ratio: minimum size of train before adjustments
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    n = len(y)

    # Initial split points
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - valid_size))

    # Step 1: Ensure all classes appear in train
    all_classes = set(np.unique(y))

    train_end = val_start

    while True:
        train_classes = set(np.unique(y[:train_end]))

        # If all classes are present → stop
        # if train_classes == all_classes: ######################### it must be back
        # if len(train_classes) >= len(all_classes) - 1:
        train_end = val_start

        # Otherwise, expand train forward
        train_end += 1

        # Safety check (avoid infinite loop)
        if train_end >= n:
            raise ValueError("Cannot ensure all classes appear in training set")

    # Step 2: Define splits
    val_end = test_start
    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    # Step 3: Logging
    logger.info("\n✓ Time-series strict split:")
    logger.info(f"Train: {len(X_train)}")
    logger.info(f"Val:   {len(X_val)}")
    logger.info(f"Test:  {len(X_test)}")

    # Optional: check distributions
    def dist(name, y_part):
        unique, counts = np.unique(y_part, return_counts=True)
        logger.info(f"{name} distribution: {dict(zip(unique, counts))}")

    dist("Train", y_train)
    dist("Val", y_val)
    dist("Test", y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
# ========================================================
def split_per_class_time_series(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Per-class chronological split:
    - 70% oldest → train
    - 15% → val
    - 15% newest → test
    """

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    classes = np.unique(y)

    for cls in classes:
        idx = np.where(y == cls)[0]   # indices of this class (already time-ordered)

        n = len(idx)

        if n == 0:
            continue

        # Compute split sizes
        n_train = max(1, int(n * train_ratio))  # ensure at least 1
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        # Edge case fix
        if n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])

        X_val_list.append(X[val_idx])
        y_val_list.append(y[val_idx])

        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])

        logger.info(f"Class {cls}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Concatenate all classes
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    return X_train, X_val, X_test, y_train, y_val, y_test

# =========================================================
# 3) DATA SHAPE CHECK
# =========================================================
def check_data_shapes(X, y, expected_dims=3):

    """
    Verify data shapes
    
    Args:
        X: Feature array
        y: Label array
        expected_dims: Expected number of dimensions for X
    """

    logger.info(f"\nData shape validation:")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    if len(X.shape) != expected_dims:
        raise ValueError(f"❌ X must be {expected_dims}D")

    if X.shape[0] != y.shape[0]:
        raise ValueError("❌ X and y mismatch")

    logger.info("✓ Shapes OK")
# ========================================================
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

# =========================================================
# 4) DEVICE
# =========================================================
def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    return device

#=========================================================
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
# =========================================================
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
# =========================================================
# 5) MAIN (EXAMPLE USAGE)
# =========================================================


# =================================================
# =========================================================
# 5) MAIN (EXAMPLE USAGE)
# =========================================================
if __name__ == "__main__":

    import sys

    # -------------------- CONFIG --------------------
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "outputs" / "data"

    X_path = DATA_PATH / "X_windows.npy"
    y_path = DATA_PATH / "y_windows.npy"

    TEST_SIZE = 0.15
    VALID_SIZE = 0.15
    RANDOM_STATE = 42

    # -------------------- LOGGING --------------------
    logger = setup_logging()

    logger.info("\n" + "█" * 80)
    logger.info("█" + " " * 20 + "STEP 4: TIME-SERIES DATA SPLITTING" + " " * 20 + "█")
    logger.info("█" * 80)

    # -------------------- SEED --------------------
    set_seed(RANDOM_STATE)

    # -------------------- LOAD DATA --------------------
    if not X_path.exists() or not y_path.exists():
        logger.error("❌ X_windows or y_windows not found!")
        logger.error("Run feature_engineering.py first")
        sys.exit(1)

    logger.info(f"📂 Loading X from: {X_path}")
    logger.info(f"📂 Loading y from: {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    logger.info(f"✓ Loaded X shape: {X.shape}")
    logger.info(f"✓ Loaded y shape: {y.shape}")

    #=========== Removing rare classes (optional, but can help with very imbalanced datasets) ============
    # =========== must omit just for training, not for test/val (to avoid data leakage) =============
    """
    unique, counts = np.unique(y, return_counts=True)

    valid_classes = unique[counts > 10]

    mask = np.isin(y, valid_classes)

    X = X[mask]
    y = y[mask]

    logger.info(f"✓ Removed rare classes")
    logger.info(f"Remaining classes: {np.unique(y)}")
    """
    """
    mask = y != 4

    X = X[mask]
    y = y[mask]

    logger.info("✓ Removed class 4 completely")
    logger.info(f"Remaining classes: {np.unique(y)}")
    """
    """
      unique_classes = np.unique(y)
    valid_classes = []

    for cls in unique_classes:
        first_idx = np.where(y == cls)[0][0]

        # keep only classes that appear early (before 70% of timeline)
        if first_idx < len(y) * 0.7:
            valid_classes.append(cls)

    mask = np.isin(y, valid_classes)

    X = X[mask]
    y = y[mask]

    logger.info("✓ Removed late-appearing classes")
    logger.info(f"Remaining classes: {np.unique(y)}")
    """
  

    # -------------------- SPLIT DATA --------------------
    # 🔥 BEST choice for your project (no leakage + all classes in train)
    """
      X_train, X_val, X_test, y_train, y_val, y_test = split_time_series_strict(
        X, y,
        valid_size=VALID_SIZE,
        test_size=TEST_SIZE
    )
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_per_class_time_series(
        X, y,
        train_ratio=0.7,
        val_ratio=0.15
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
    logger.info(" - X_train.npy, X_val.npy, X_test.npy")
    logger.info(" - y_train.npy, y_val.npy, y_test.npy")

    logger.info("\n✅ STEP 4 COMPLETE\n")