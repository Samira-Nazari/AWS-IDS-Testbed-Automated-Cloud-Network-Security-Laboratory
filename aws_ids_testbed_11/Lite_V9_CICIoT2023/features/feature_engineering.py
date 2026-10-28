"""
Feature Engineering module for CIC-DDoS2019
Handles feature selection and sliding window creation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
"""
Model used to:
learn patterns
compute feature importance
"""
from sklearn.feature_selection import SelectFromModel
"""
Used to:
automatically select important features
"""
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
"""For evaluation:
cross_val_score → test model performance
TimeSeriesSplit → special split for time-series (no shuffling)"""
import logging
import os

logger = logging.getLogger(__name__)


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _balanced_window_subset_for_rf(X_windows_flat: np.ndarray,
                                   y_windows: np.ndarray,
                                   max_windows_per_label: int,
                                   random_state: int) -> tuple:
    """
    Build a balanced, memory-safe subset for RandomForest window selection.

    This affects only RF scoring/feature importance. Final train/val/test
    windows are still created later from all valid rows.
    """
    if not max_windows_per_label or max_windows_per_label <= 0:
        return X_windows_flat, y_windows

    rng = np.random.default_rng(random_state)
    selected_indices = []

    for label in np.unique(y_windows):
        label_indices = np.where(y_windows == label)[0]

        if len(label_indices) > max_windows_per_label:
            chosen = rng.choice(
                label_indices,
                size=max_windows_per_label,
                replace=False
            )
        else:
            chosen = label_indices

        selected_indices.append(chosen)

    selected_indices = np.concatenate(selected_indices)
    selected_indices.sort()

    return X_windows_flat[selected_indices], y_windows[selected_indices]


def _balanced_window_subset_for_rf_streaming(
        X: np.ndarray,
        y: np.ndarray,
        window_size: int,
        step_size: int,
        max_windows_per_label: int,
        random_state: int,
        batch_window_starts: int = 20000,
    ) -> tuple:
    """
    Build a balanced, balanced subset for RF using streamed windows.

    This avoids creating the full sliding-window matrix in memory for large
    window configurations by generating windows in smaller start-index batches.
    """
    if not max_windows_per_label or max_windows_per_label <= 0:
        raise ValueError(
            "Streaming RF subset requires a finite max_windows_per_label"
        )

    n_samples, n_features = X.shape
    if n_samples < window_size:
        return np.empty((0, window_size * n_features), dtype=np.float32), np.empty((0,), dtype=y.dtype)

    all_labels = np.unique(y)
    label_to_slot = {label: idx for idx, label in enumerate(all_labels)}
    num_labels = len(all_labels)

    # Preallocate reservoir arrays for each label
    reservoir = [
        np.empty((max_windows_per_label, window_size * n_features), dtype=np.float32)
        for _ in range(num_labels)
    ]
    counts = np.zeros(num_labels, dtype=np.int64)
    seen = np.zeros(num_labels, dtype=np.int64)

    starts = np.arange(0, n_samples - window_size + 1, step_size, dtype=np.int64)
    rng = np.random.default_rng(random_state)

    for batch_start in range(0, len(starts), batch_window_starts):
        batch_starts = starts[batch_start:batch_start + batch_window_starts]
        segment_start = int(batch_starts[0])
        segment_end = int(batch_starts[-1] + window_size)
        chunk = X[segment_start:segment_end]

        windows = FeatureEngineer.create_sliding_window(chunk, window_size, step_size)
        if windows.size == 0:
            continue

        X_flat = windows.reshape((windows.shape[0], -1)).astype(np.float32, copy=False)
        y_batch = y[batch_starts + window_size - 1]

        for row_idx, label in enumerate(y_batch):
            slot = label_to_slot[label]
            current_seen = seen[slot]
            if counts[slot] < max_windows_per_label:
                reservoir[slot][counts[slot]] = X_flat[row_idx]
                counts[slot] += 1
            else:
                replace_index = rng.integers(0, current_seen + 1)
                if replace_index < max_windows_per_label:
                    reservoir[slot][replace_index] = X_flat[row_idx]
            seen[slot] += 1

    X_samples = []
    y_samples = []
    for label, slot in label_to_slot.items():
        if counts[slot] > 0:
            X_samples.append(reservoir[slot][:counts[slot]])
            y_samples.append(np.full(counts[slot], label, dtype=y.dtype))

    if not X_samples:
        return np.empty((0, window_size * n_features), dtype=np.float32), np.empty((0,), dtype=y.dtype)

    X_out = np.concatenate(X_samples, axis=0)
    y_out = np.concatenate(y_samples, axis=0)

    order = np.argsort(y_out)
    return X_out[order], y_out[order]


class FeatureEngineer:
    """Handle feature selection and engineering"""
    # A class that groups all feature engineering logic.
    
    def __init__(self, window_sizes: list, step_sizes: list, 
                 rf_n_estimators: int = 100, cv_splits: int = 5):
        """
        Initialize feature engineer
        
        Args:
            window_sizes: List of window sizes to try
            step_sizes: List of step sizes to try
            rf_n_estimators: Number of RF estimators
            cv_splits: Number of cross-validation splits
        """
        """
        window_sizes → e.g. [5, 10, 20]
        step_sizes → e.g. [1, 2, 5]
        rf_n_estimators → number of trees in RandomForest
        cv_splits → number of folds in cross-validation
        """
        self.window_sizes = window_sizes
        self.step_sizes = step_sizes
        self.rf_n_estimators = rf_n_estimators
        self.cv_splits = cv_splits
        
        self.best_window_size = None
        self.best_step_size = None
        self.selected_features = None
        self.feature_importances = None
    

    """
    @staticmethod in Python is a decorator that defines a method inside a class 
    which does not depend on the instance (self) or the class (cls). Unlike regular methods 
    that use object data or class methods that use class-level data, 
    a static method acts like a normal standalone function but is placed inside the class 
    for logical organization. It’s typically used for utility functions 
    that are related to the class conceptually but don’t need access to its internal state."""
    @staticmethod
    def create_sliding_window(data: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """
        Create sliding windows using a memory-safer NumPy view before final copy.
        
        Args:
            data: Input data array
            window_size: Size of each window
            step_size: Step size between windows
            
        Returns:
            Array of sliding windows
        """
        if len(data) < window_size:
            return np.empty((0, window_size) + data.shape[1:], dtype=data.dtype)

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(data, window_shape=window_size, axis=0)

        if data.ndim > 1:
            windows = np.moveaxis(windows, -1, 1)

        return np.ascontiguousarray(windows[::step_size])
    
    def select_features(self, df: pd.DataFrame, target_column: str = 'Label') -> tuple:
        """
        Select features using RandomForest with sliding windows
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (X_windows, y_windows, selected_features, columns_to_drop)
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: FEATURE ENGINEERING & SELECTION")
        logger.info("="*60)
        
        logger.info(f"\nInitial features: {df.shape[1] - 1}")  # Exclude target
        
        # Separate features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        logger.info(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

        from config.config import USE_ALL_FEATURES

        valid_configs = [
            (window_size, step_size)
            for window_size in self.window_sizes
            for step_size in self.step_sizes
            if window_size <= len(X) and step_size <= window_size
        ]

        if not valid_configs:
            raise ValueError("Feature selection failed: no valid window configuration was found.")

        if USE_ALL_FEATURES and len(valid_configs) == 1:
            self.best_window_size, self.best_step_size = valid_configs[0]
            selected_feature_names = df.drop(columns=[target_column]).columns.tolist()
            columns_to_drop = []

            self.selected_features = selected_feature_names
            self.feature_importances = np.ones(len(selected_feature_names), dtype=np.float32)

            logger.info("USE_ALL_FEATURES=True and only one window config is active.")
            logger.info("Skipping RandomForest window scoring to avoid huge memory use.")
            logger.info(f"Best window size: {self.best_window_size}")
            logger.info(f"Best step size: {self.best_step_size}")
            logger.info(f"Selected features: {len(selected_feature_names)}")

            return (
                np.empty(
                    (0, self.best_window_size, len(selected_feature_names)),
                    dtype=np.float32,
                ),
                np.empty((0,), dtype=y.dtype),
                selected_feature_names,
                columns_to_drop,
            )
        
        # Start with worst score
        best_score = -np.inf
        best_model = None
        best_importances = None
        best_selected_features = None
        
        logger.info("\n🔍 Testing window configurations...")
        logger.info(f"   Window sizes: {self.window_sizes}")
        logger.info(f"   Step sizes: {self.step_sizes}")
        logger.info(f"   Cross-validation splits: {self.cv_splits}")
        
        config_count = 0
        for window_size in self.window_sizes:
            for step_size in self.step_sizes:
                # Skip invalid configurations
                if window_size > len(X) or step_size > window_size:
                    continue
                
                config_count += 1
                
                # Generate balanced RF sample windows via streaming, without building the
                # full sliding-window matrix in memory.
                from config.config import RF_MAX_WINDOWS_PER_LABEL, RF_RANDOM_STATE

                X_rf_flat, y_rf = _balanced_window_subset_for_rf_streaming(
                    X,
                    y,
                    window_size=window_size,
                    step_size=step_size,
                    max_windows_per_label=RF_MAX_WINDOWS_PER_LABEL,
                    random_state=RF_RANDOM_STATE,
                    batch_window_starts=20000,
                )

                total_windows = max(0, (len(X) - window_size) // step_size + 1)

                logger.info(f"\n   [{config_count}] Window: {window_size}, Step: {step_size}")
                logger.info(f"       Total sliding windows: {total_windows}")
                logger.info(f"       Window feature length: {window_size * X.shape[1]}")
                logger.info(f"       RF subset samples: {X_rf_flat.shape[0]}")
                
                # Train RandomForest
                logger.info(f"       Training RandomForest with {self.rf_n_estimators} estimators...")
                from config.config import RF_N_JOBS

                model = RandomForestClassifier(
                    n_estimators=self.rf_n_estimators,
                    random_state=42,
                    n_jobs=RF_N_JOBS,
                )
                # Use a serialized RF build to reduce memory pressure during cross-validation

                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                # Creates time-series cross-validation splits. It keeps time order and does not shuffle.
                # A special way to split data without breaking time order
                # Testing uses future data
                scores = cross_val_score(model, X_rf_flat, y_rf, cv=tscv,
                                        scoring='accuracy', n_jobs=1)
                # Trains and tests the model several times, then returns accuracy scores.
                avg_score = scores.mean()
                
                logger.info(f"       CV Scores: {[f'{s:.4f}' for s in scores]}")
                logger.info(f"       Mean CV Score: {avg_score:.6f} (±{scores.std():.6f})")
                
                # Check if this is the best configuration
                if avg_score > best_score:
                    logger.info(f"       ✓ New best score! (was {best_score:.6f})")
                    best_score = avg_score
                    self.best_window_size = window_size
                    self.best_step_size = step_size
                    
                    # Fit model on all data
                    # Trains the RandomForest on all windowed data using the best configuration.
                    model.fit(X_rf_flat, y_rf)
                    """
                    cross_val_score → "How good is my model?"
                    model.fit       → Train final model using everything
                    """
                    best_model = model
                    
                    # Extract feature importances
                    importances = model.feature_importances_
                    
                    # Average importances across windows for each original feature
                    n_features = X.shape[1]
                    importances_reshaped = importances.reshape((window_size, n_features))
                    # Converts importances back into: (window_size, original_features)
                    averaged_importances = np.mean(importances_reshaped, axis=0)
                    # Averages importance over time steps, so each original feature gets one importance score
                    best_importances = averaged_importances
                    # Uses average importance as the cutoff threshold.
                    
                    # Feature selection
                    threshold = np.mean(averaged_importances)
                    selector = SelectFromModel(model, threshold=threshold, prefit=True)
                    # Creates a selector to keep only important features.
                    # prefit=True, The model is already trained — don't fit it again
                    best_selected_features = selector.get_support(indices=True)
                    # Gets indexes of selected features.
        
        if best_selected_features is None or best_importances is None:
            raise ValueError("Feature selection failed: no valid window configuration was found.")

        # Extract selected feature names
        feature_names = df.drop(columns=[target_column]).columns
        # It creates a list of feature (input) column names, excluding the target (label).
        if USE_ALL_FEATURES:
            selected_feature_names = feature_names.tolist()
            columns_to_drop = []
            logger.info("USE_ALL_FEATURES=True, keeping every input feature after window selection")
        else:
            selected_feature_set = set(feature_names[best_selected_features % len(feature_names)])
            selected_feature_names = [col for col in feature_names if col in selected_feature_set]
            # You are mapping indices from flattened sliding windows back to the original feature names.
            columns_to_drop = [col for col in feature_names if col not in selected_feature_set]

        logger.info(f"selected_feature_names: {selected_feature_names}")

        if not selected_feature_names:
            raise ValueError("Feature selection failed: no input features were selected.")
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE SELECTION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Best window size: {self.best_window_size}")
        logger.info(f"Best step size: {self.best_step_size}")
        logger.info(f"Best cross-validation score: {best_score:.6f}")
        logger.info(f"\nFeature importance threshold: {np.mean(best_importances):.6f}")
        logger.info(f"Selected features: {len(selected_feature_names)} / {len(feature_names)}")
        logger.info(f"Dropped features: {len(columns_to_drop)}")
        
        logger.info(f"\nTop 10 most important features:")
        top_indices = np.argsort(best_importances)[::-1][:10]
        # top_indices = indices of most important features
        # np.argsort(...) Returns indices sorted from smallest to largest:
        # [::-1] Reverses it → largest first:
        for rank, idx in enumerate(top_indices, 1):
            logger.info(f"  {rank:2d}. {feature_names[idx]:30s}: {best_importances[idx]:.6f}")
            # enumerate(..., 1) means you loop through items while counting them starting from 1 instead of 0.
        if columns_to_drop:
            logger.info(f"\nFeatures to drop ({len(columns_to_drop)}):")
            for col in columns_to_drop[:5]:
                logger.info(f"  - {col}")
            if len(columns_to_drop) > 5:
                logger.info(f"  ... and {len(columns_to_drop) - 5} more")
        
        # Store for later use
        self.selected_features = selected_feature_names
        self.feature_importances = best_importances
        
        # Create final sliding windows with only the selected features.
        X_selected = df[selected_feature_names].values
        X_windows_best = self.create_sliding_window(X_selected, self.best_window_size, self.best_step_size)
        y_windows_best = self.create_sliding_window(y, self.best_window_size, self.best_step_size)[:, -1]
        
        logger.info(f"\nFinal sliding windows:")
        logger.info(f"  X shape: {X_windows_best.shape}")
        logger.info(f"  y shape: {y_windows_best.shape}")
        logger.info(f"  (Samples, Window size, Features) = {X_windows_best.shape}")
        logger.info(f"  Final model features: {len(selected_feature_names)}")
        
        logger.info("\n✓ Feature engineering complete!")
        logger.info("="*60 + "\n")
        
        return X_windows_best, y_windows_best, selected_feature_names, columns_to_drop


def engineer_features(df: pd.DataFrame, window_sizes: list, step_sizes: list,
                     target_column: str = 'Label') -> tuple:
    """
    Main function for feature engineering
    
    Args:
        df: Input DataFrame
        window_sizes: Sizes to try
        step_sizes: Step sizes to try
        target_column: Target column name
        
    Returns:
        Tuple of (X_windows, y_windows, selected_features, columns_to_drop)
    """
    engineer = FeatureEngineer(window_sizes, step_sizes)
    return engineer.select_features(df, target_column)


def _prepare_modeling_dataframe(df: pd.DataFrame,
                                target_column: str = 'Label') -> pd.DataFrame:
    """
    Keep only numeric model inputs and the target.

    Timestamp and identifiers may be useful for ordering/splitting, but they
    should not be model features.
    """
    df = df.drop(
        columns=[
            "Source IP",
            "Destination IP",
            "SimillarHTTP",
            "Flow ID",
            "unique_id",
            "Timestamp",
            "planned_split",
            "split_segment_id",
            "source_row_index",
        ],
        errors='ignore',
    )
    df = df.select_dtypes(include=[np.number])

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was lost during preprocessing.")

    return df


def _split_per_class_time_series_dataframe(df: pd.DataFrame,
                                           label_column: str = 'Label',
                                           timestamp_column: str = 'Timestamp',
                                           valid_size: float = 0.2,
                                           test_size: float = 0.2) -> tuple:
    if "planned_split" in df.columns:
        required_columns = {
            label_column,
            "unique_id",
            "planned_split",
            "split_segment_id",
            "source_row_index",
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                "Capture-planned data is missing required metadata columns: "
                f"{sorted(missing_columns)}"
            )

        allowed_splits = {"train", "validation", "test"}
        observed_splits = set(df["planned_split"].dropna().unique())
        unexpected_splits = observed_splits - allowed_splits
        if unexpected_splits:
            raise ValueError(
                f"Unexpected planned_split values: {sorted(unexpected_splits)}"
            )
        missing_splits = allowed_splits - observed_splits
        if missing_splits:
            raise ValueError(
                f"Capture split plan is missing splits: {sorted(missing_splits)}"
            )

        duplicate_source_rows = df.duplicated(
            subset=["unique_id", "source_row_index"], keep=False
        )
        if duplicate_source_rows.any():
            raise ValueError(
                "Capture-planned input contains source rows in more than one split"
            )

        split_frames = {}
        for split in ["train", "validation", "test"]:
            split_df = df[df["planned_split"] == split].copy()
            sort_columns = [
                column
                for column in [
                    label_column,
                    "unique_id",
                    "split_segment_id",
                    "source_row_index",
                ]
                if column in split_df.columns
            ]
            split_frames[split] = split_df.sort_values(
                sort_columns, kind="stable"
            ).reset_index(drop=True)
            logger.info(
                "Planned split %s: %d rows across %d segments",
                split,
                len(split_frames[split]),
                split_frames[split]["split_segment_id"].nunique(),
            )

        return (
            split_frames["train"],
            split_frames["validation"],
            split_frames["test"],
        )

    train_ratio = 1.0 - valid_size - test_size
    if train_ratio <= 0:
        raise ValueError("Invalid split sizes: train ratio must be positive.")

    if "unique_id" in df.columns:
        synthetic_mask = df["unique_id"].astype(str).str.startswith("synth_")
    else:
        synthetic_mask = pd.Series(False, index=df.index)

    synthetic_df = df[synthetic_mask].copy()
    real_df = df[~synthetic_mask].copy()

    if len(synthetic_df) > 0:
        logger.info(
            "Detected %d synthetic rows; keeping them in train split only",
            len(synthetic_df)
        )

    if timestamp_column in df.columns:
        df_ordered = real_df.sort_values(timestamp_column).reset_index(drop=True)
    else:
        logger.warning("Timestamp column not found; using existing row order")
        df_ordered = real_df.reset_index(drop=True)

    train_parts = []
    val_parts = []
    test_parts = []

    for cls in df_ordered[label_column].dropna().unique():
        class_df = df_ordered[df_ordered[label_column] == cls].copy()
        if timestamp_column in class_df.columns:
            class_df = class_df.sort_values(timestamp_column)

        n = len(class_df)
        n_train = max(1, int(n * train_ratio))
        n_val = int(n * valid_size)
        n_test = n - n_train - n_val

        if n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        train_parts.append(class_df.iloc[:n_train])
        val_parts.append(class_df.iloc[n_train:n_train + n_val])
        test_parts.append(class_df.iloc[n_train + n_val:])

        logger.info(f"Class {cls}: Train={n_train}, Val={n_val}, Test={n_test}")

    if len(synthetic_df) > 0:
        train_parts.append(synthetic_df)
        synthetic_counts = synthetic_df[label_column].value_counts().sort_index()
        for cls, count in synthetic_counts.items():
            logger.info(f"Class {cls}: Synthetic train-only={count}")

    train_df = pd.concat(train_parts, axis=0)
    val_df = pd.concat(val_parts, axis=0)
    test_df = pd.concat(test_parts, axis=0)

    if timestamp_column in df.columns:
        train_df = train_df.sort_values(timestamp_column).reset_index(drop=True)
        val_df = val_df.sort_values(timestamp_column).reset_index(drop=True)
        test_df = test_df.sort_values(timestamp_column).reset_index(drop=True)
    else:
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def _create_split_windows(engineer: FeatureEngineer,
                          X_values: np.ndarray,
                          y_values: np.ndarray) -> tuple:
    X_windows = engineer.create_sliding_window(
        X_values,
        engineer.best_window_size,
        engineer.best_step_size
    )
    y_windows = engineer.create_sliding_window(
        y_values,
        engineer.best_window_size,
        engineer.best_step_size
    )[:, -1]
    return X_windows, y_windows


def _create_split_windows_by_segment(engineer: FeatureEngineer,
                                     split_df: pd.DataFrame,
                                     scaled_features: np.ndarray,
                                     y_values: np.ndarray,
                                     split_name: str,
                                     label_column: str = 'Label',
                                     timestamp_column: str = 'Timestamp') -> tuple:
    """
    Create real sliding windows separately inside each capture/split segment.

    The split dataframe must already be train, validation, or test. This function
    prevents windows from crossing label, CSV capture, or split boundaries.
    """
    X_parts = []
    y_parts = []
    audit_rows = []

    working_df = split_df.reset_index(drop=True).copy()
    working_df["_row_pos"] = np.arange(len(working_df))

    required_columns = {
        label_column,
        "unique_id",
        "planned_split",
        "split_segment_id",
        "source_row_index",
    }
    missing_columns = required_columns - set(working_df.columns)
    if missing_columns:
        raise ValueError(
            "Capture-safe windowing requires metadata columns: "
            f"{sorted(missing_columns)}"
        )

    group_columns = [label_column, "unique_id", "split_segment_id"]
    for group_key, segment_df in working_df.groupby(
        group_columns, sort=False, dropna=False
    ):
        label, unique_id, split_segment_id = group_key
        segment_df = segment_df.sort_values(
            ["source_row_index", timestamp_column], kind="stable"
        )

        if segment_df[label_column].nunique() != 1:
            raise AssertionError(
                f"Window segment {split_segment_id} contains multiple labels"
            )
        if segment_df["unique_id"].nunique() != 1:
            raise AssertionError(
                f"Window segment {split_segment_id} contains multiple unique_id values"
            )
        if segment_df["planned_split"].nunique() != 1:
            raise AssertionError(
                f"Window segment {split_segment_id} contains multiple splits"
            )
        actual_split = str(segment_df["planned_split"].iloc[0])
        if actual_split != split_name:
            raise AssertionError(
                f"Window segment {split_segment_id} belongs to {actual_split}, "
                f"not {split_name}"
            )

        source_rows = segment_df["source_row_index"].to_numpy(dtype=np.int64)
        if len(source_rows) > 1 and np.any(np.diff(source_rows) != 1):
            raise AssertionError(
                f"Window segment {split_segment_id} is not source-row contiguous"
            )

        row_positions = segment_df["_row_pos"].to_numpy()
        segment_row_count = len(row_positions)
        expected_windows = max(
            0,
            (segment_row_count - engineer.best_window_size)
            // engineer.best_step_size
            + 1,
        )

        if len(row_positions) < engineer.best_window_size:
            logger.info(
                f"Skipping segment {split_segment_id}: only {len(row_positions)} "
                f"rows, needs at least {engineer.best_window_size}"
            )
            audit_rows.append(
                {
                    "split": split_name,
                    "Label": int(label),
                    "unique_id": str(unique_id),
                    "split_segment_id": str(split_segment_id),
                    "source_start_row": int(source_rows.min()),
                    "source_end_row": int(source_rows.max()) + 1,
                    "segment_row_count": segment_row_count,
                    "expected_windows": expected_windows,
                    "actual_windows": 0,
                    "is_synthetic": False,
                }
            )
            continue

        X_segment = scaled_features[row_positions]
        y_segment = y_values[row_positions]

        X_windows, y_windows = _create_split_windows(
            engineer,
            X_segment,
            y_segment
        )

        if len(y_windows) != expected_windows:
            raise AssertionError(
                f"Window count mismatch for {split_segment_id}: expected "
                f"{expected_windows}, got {len(y_windows)}"
            )
        if len(y_windows) and not np.all(y_windows == int(label)):
            raise AssertionError(
                f"Generated windows for {split_segment_id} contain mixed labels"
            )

        X_parts.append(X_windows)
        y_parts.append(y_windows)
        audit_rows.append(
            {
                "split": split_name,
                "Label": int(label),
                "unique_id": str(unique_id),
                "split_segment_id": str(split_segment_id),
                "source_start_row": int(source_rows.min()),
                "source_end_row": int(source_rows.max()) + 1,
                "segment_row_count": segment_row_count,
                "expected_windows": expected_windows,
                "actual_windows": len(y_windows),
                "is_synthetic": False,
            }
        )

        logger.info(
            f"Segment {split_segment_id}: created {len(y_windows)} windows "
            f"from {len(row_positions)} rows"
        )

    if not X_parts:
        raise ValueError(
            "No valid windows were created. Check window size and per-label row counts."
        )

    audit_df = pd.DataFrame.from_records(audit_rows)
    if int(audit_df["actual_windows"].sum()) != sum(len(part) for part in y_parts):
        raise AssertionError(f"Window audit total mismatch for split {split_name}")

    logger.info(
        "Capture-safe %s windows: %d windows from %d real segments",
        split_name,
        int(audit_df["actual_windows"].sum()),
        len(audit_df),
    )
    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        audit_df,
    )


def _augment_train_windows_with_smote_enn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_per_class: int,
    min_count: int,
    k_neighbors: int,
    enn_neighbors: int,
    enn_min_agreement: float,
    random_state: int,
    preserve_original_majorities: bool = True,
) -> tuple:
    """
    Apply SMOTE followed by ENN-style cleaning to complete train windows only.

    Validation/test windows remain real. Original train windows are preserved by
    default; ENN is used only to filter synthetic windows that are inconsistent
    with their local neighborhood.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y_train, return_counts=True)
    counts_map = dict(zip(classes.astype(int), counts.astype(int)))
    target_classes = [
        cls for cls, count in counts_map.items()
        if count < min_count and count < target_per_class
    ]

    audit_by_class = {
        int(cls): {
            "class_id": int(cls),
            "real_train_windows": int(count),
            "targeted_for_smote": int(cls) in target_classes,
            "smote_target_per_class": int(target_per_class),
            "smote_generated": 0,
            "enn_retained": 0,
            "enn_removed": 0,
            "final_train_windows": int(count),
            "status": "targeted" if int(cls) in target_classes else "not_targeted",
        }
        for cls, count in counts_map.items()
    }

    def finalize_audit(final_labels: np.ndarray) -> pd.DataFrame:
        final_classes, final_counts = np.unique(final_labels, return_counts=True)
        final_counts_map = dict(
            zip(final_classes.astype(int), final_counts.astype(int))
        )
        for class_id, audit_row in audit_by_class.items():
            audit_row["final_train_windows"] = int(
                final_counts_map.get(class_id, 0)
            )
        return pd.DataFrame.from_records(
            [audit_by_class[class_id] for class_id in sorted(audit_by_class)]
        )

    if not target_classes:
        logger.info("Window-level SMOTE-ENN: no minority train-window classes found")
        return X_train, y_train, finalize_audit(y_train)

    logger.info("\nApplying window-level SMOTE-ENN to train windows only")
    logger.info(f"  Original train-window counts: {counts_map}")
    logger.info(f"  Target minority classes: {target_classes}")
    if preserve_original_majorities:
        logger.info("  Preserving all original train windows; ENN cleans synthetic windows only")

    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError(
            "Window-level SMOTE-ENN requires imbalanced-learn and scikit-learn. "
            "Install them or disable APPLY_SMOTE_AFTER_WINDOW."
        ) from exc

    synthetic_X_parts = []
    synthetic_y_parts = []
    window_shape = X_train.shape[1:]

    for cls in target_classes:
        cls_count = counts_map[cls]
        if cls_count < 2:
            audit_by_class[int(cls)]["status"] = "insufficient_real_windows"
            logger.warning(
                "SMOTE-ENN skipped for class %s because it has only %d train window(s)",
                cls,
                cls_count,
            )
            continue

        cls_indices = np.where(y_train == cls)[0]
        helper_indices = np.where(y_train != cls)[0]
        helper_count = min(max(cls_count, 1000), len(helper_indices))

        sampled_helper_indices = rng.choice(
            helper_indices,
            size=helper_count,
            replace=False,
        )
        subset_indices = np.concatenate([cls_indices, sampled_helper_indices])

        X_subset = X_train[subset_indices].reshape((len(subset_indices), -1))
        y_subset = y_train[subset_indices]

        effective_k = min(k_neighbors, cls_count - 1)
        smote = SMOTE(
            sampling_strategy={cls: target_per_class},
            k_neighbors=effective_k,
            random_state=random_state,
        )

        X_resampled_flat, y_resampled = smote.fit_resample(X_subset, y_subset)

        synthetic_flat = X_resampled_flat[len(subset_indices):]
        synthetic_labels = y_resampled[len(subset_indices):]

        if len(synthetic_labels) == 0:
            audit_by_class[int(cls)]["status"] = "no_synthetic_generated"
            logger.info("  Class %s: SMOTE added 0 windows", cls)
            continue

        # ENN-style cleaning for synthetic windows only.
        nn_count = min(enn_neighbors + 1, len(X_resampled_flat))
        neighbors = NearestNeighbors(n_neighbors=nn_count, n_jobs=-1)
        neighbors.fit(X_resampled_flat)
        neighbor_indices = neighbors.kneighbors(
            synthetic_flat,
            return_distance=False,
        )

        keep_mask = []
        for idxs, synth_label in zip(neighbor_indices, synthetic_labels):
            neighbor_labels = y_resampled[idxs[1:]]
            matching_neighbors = np.count_nonzero(neighbor_labels == int(synth_label))
            required_matches = int(np.ceil(enn_min_agreement * len(neighbor_labels)))
            keep_mask.append(matching_neighbors >= required_matches)

        keep_mask = np.array(keep_mask, dtype=bool)
        cleaned_synthetic_flat = synthetic_flat[keep_mask]
        cleaned_synthetic_labels = synthetic_labels[keep_mask]

        removed_count = len(synthetic_labels) - len(cleaned_synthetic_labels)
        audit_by_class[int(cls)]["smote_generated"] = int(len(synthetic_labels))
        audit_by_class[int(cls)]["enn_retained"] = int(
            len(cleaned_synthetic_labels)
        )
        audit_by_class[int(cls)]["enn_removed"] = int(removed_count)
        if len(cleaned_synthetic_labels) == 0:
            audit_by_class[int(cls)]["status"] = "all_synthetic_removed"
            logger.info(
                "  Class %s: ENN removed all %d synthetic windows",
                cls,
                len(synthetic_labels),
            )
            continue

        synthetic_X_parts.append(
            cleaned_synthetic_flat.reshape((-1, window_shape[0], window_shape[1]))
        )
        synthetic_y_parts.append(cleaned_synthetic_labels)
        audit_by_class[int(cls)]["status"] = "completed"

        logger.info(
            "  Class %s: %d -> %d target; SMOTE added %d, ENN kept %d, removed %d",
            cls,
            cls_count,
            target_per_class,
            len(synthetic_labels),
            len(cleaned_synthetic_labels),
            removed_count,
        )

    if not synthetic_X_parts:
        logger.info("Window-level SMOTE-ENN did not add any train windows")
        return X_train, y_train, finalize_audit(y_train)

    X_augmented = np.concatenate([X_train] + synthetic_X_parts, axis=0)
    y_augmented = np.concatenate([y_train] + synthetic_y_parts, axis=0)

    res_classes, res_counts = np.unique(y_augmented, return_counts=True)
    res_counts_map = dict(zip(res_classes.astype(int), res_counts.astype(int)))
    logger.info(f"  Augmented train-window counts: {res_counts_map}")
    logger.info(f"  Total synthetic train windows added: {len(y_augmented) - len(y_train)}")

    return X_augmented, y_augmented, finalize_audit(y_augmented)


def engineer_features_no_leakage(df: pd.DataFrame, window_sizes: list,
                                 step_sizes: list,
                                 target_column: str = 'Label',
                                 valid_size: float = 0.2,
                                 test_size: float = 0.2,
                                 rf_n_estimators: int = 100,
                                 cv_splits: int = 5) -> dict:
    """
    Leakage-safe feature engineering.

    Order:
    1. encode labels
    2. split dataframe per class chronologically
    3. fit RandomForest feature selection on train only
    4. fit StandardScaler on selected train features only
    5. transform validation/test
    6. create windows separately for each split and label
    7. optionally add SMOTE-ENN synthetic complete windows to train only
    """
    logger.info("\n" + "="*60)
    logger.info("LEAKAGE-SAFE FEATURE ENGINEERING")
    logger.info("="*60)

    df = df.copy()

    required_capture_metadata = {
        "unique_id",
        "planned_split",
        "split_segment_id",
        "source_row_index",
    }
    missing_capture_metadata = required_capture_metadata - set(df.columns)
    if missing_capture_metadata:
        raise ValueError(
            "Lite_V9 capture-safe feature engineering requires Step 1-3 "
            f"metadata: {sorted(missing_capture_metadata)}"
        )

    train_df, val_df, test_df = _split_per_class_time_series_dataframe(
        df,
        label_column=target_column,
        timestamp_column='Timestamp',
        valid_size=valid_size,
        test_size=test_size
    )

    label_classes = pd.Index(train_df[target_column].drop_duplicates())
    label_to_id = {label: idx for idx, label in enumerate(label_classes)}

    for split_name, split_df in [
        ("validation", val_df),
        ("test", test_df),
    ]:
        unseen_labels = set(split_df[target_column].dropna().unique()) - set(label_to_id)
        if unseen_labels:
            raise ValueError(
                f"{split_name} split contains labels not present in train: "
                f"{sorted(unseen_labels)}"
            )

    train_df[target_column] = train_df[target_column].map(label_to_id).astype(int)
    val_df[target_column] = val_df[target_column].map(label_to_id).astype(int)
    test_df[target_column] = test_df[target_column].map(label_to_id).astype(int)

    train_model_df = _prepare_modeling_dataframe(train_df, target_column)
    val_model_df = _prepare_modeling_dataframe(val_df, target_column)
    test_model_df = _prepare_modeling_dataframe(test_df, target_column)

    engineer = FeatureEngineer(
        window_sizes,
        step_sizes,
        rf_n_estimators=rf_n_estimators,
        cv_splits=cv_splits
    )

    _, _, selected_features, columns_to_drop = engineer.select_features(
        train_model_df,
        target_column=target_column
    )

    scaler = StandardScaler()
    scaler.fit(train_model_df[selected_features])

    X_train_scaled = scaler.transform(
        train_model_df[selected_features]
    ).astype(np.float32, copy=False)
    X_val_scaled = scaler.transform(
        val_model_df[selected_features]
    ).astype(np.float32, copy=False)
    X_test_scaled = scaler.transform(
        test_model_df[selected_features]
    ).astype(np.float32, copy=False)

    y_train_values = train_model_df[target_column].to_numpy()
    y_val_values = val_model_df[target_column].to_numpy()
    y_test_values = test_model_df[target_column].to_numpy()

    X_train, y_train, train_window_audit = _create_split_windows_by_segment(
        engineer,
        train_df,
        X_train_scaled,
        y_train_values,
        split_name="train",
        label_column=target_column,
        timestamp_column='Timestamp'
    )
    X_val, y_val, val_window_audit = _create_split_windows_by_segment(
        engineer,
        val_df,
        X_val_scaled,
        y_val_values,
        split_name="validation",
        label_column=target_column,
        timestamp_column='Timestamp'
    )
    X_test, y_test, test_window_audit = _create_split_windows_by_segment(
        engineer,
        test_df,
        X_test_scaled,
        y_test_values,
        split_name="test",
        label_column=target_column,
        timestamp_column='Timestamp'
    )

    real_train_windows = len(y_train)
    real_val_windows = len(y_val)
    real_test_windows = len(y_test)

    from config.config import (
        APPLY_SMOTE_AFTER_WINDOW,
        SMOTE_ENN_MIN_AGREEMENT,
        SMOTE_ENN_NEIGHBORS,
        SMOTE_K_NEIGHBORS,
        SMOTE_MIN_COUNT,
        SMOTE_PRESERVE_ORIGINAL_MAJORITIES,
        SMOTE_RANDOM_STATE,
        SMOTE_TARGET_PER_CLASS,
    )

    if APPLY_SMOTE_AFTER_WINDOW:
        X_train, y_train, smote_audit = _augment_train_windows_with_smote_enn(
            X_train,
            y_train,
            target_per_class=SMOTE_TARGET_PER_CLASS,
            min_count=SMOTE_MIN_COUNT,
            k_neighbors=SMOTE_K_NEIGHBORS,
            enn_neighbors=SMOTE_ENN_NEIGHBORS,
            enn_min_agreement=SMOTE_ENN_MIN_AGREEMENT,
            random_state=SMOTE_RANDOM_STATE,
            preserve_original_majorities=SMOTE_PRESERVE_ORIGINAL_MAJORITIES,
        )
    else:
        classes, counts = np.unique(y_train, return_counts=True)
        smote_audit = pd.DataFrame(
            {
                "class_id": classes.astype(int),
                "real_train_windows": counts.astype(int),
                "targeted_for_smote": False,
                "smote_target_per_class": SMOTE_TARGET_PER_CLASS,
                "smote_generated": 0,
                "enn_retained": 0,
                "enn_removed": 0,
                "final_train_windows": counts.astype(int),
                "status": "smote_disabled",
            }
        )

    synthetic_train_windows = len(y_train) - real_train_windows
    if len(y_val) != real_val_windows or len(y_test) != real_test_windows:
        raise AssertionError(
            "Validation or test windows changed during training-only SMOTE"
        )

    class_names = list(label_classes)
    smote_audit.insert(
        1,
        "class_name",
        smote_audit["class_id"].map(
            lambda class_id: class_names[int(class_id)]
        ),
    )
    generated_total = int(smote_audit["smote_generated"].sum())
    retained_total = int(smote_audit["enn_retained"].sum())
    removed_total = int(smote_audit["enn_removed"].sum())
    if generated_total != retained_total + removed_total:
        raise AssertionError(
            "SMOTE audit mismatch: generated does not equal retained plus removed"
        )
    if retained_total != synthetic_train_windows:
        raise AssertionError(
            "SMOTE audit retained total does not match appended training windows"
        )

    final_classes, final_counts = np.unique(y_train, return_counts=True)
    final_counts_map = dict(
        zip(final_classes.astype(int), final_counts.astype(int))
    )
    for audit_row in smote_audit.itertuples(index=False):
        if int(audit_row.final_train_windows) != int(
            final_counts_map.get(int(audit_row.class_id), 0)
        ):
            raise AssertionError(
                f"SMOTE audit final count mismatch for class {audit_row.class_id}"
            )

    window_segment_audit = pd.concat(
        [train_window_audit, val_window_audit, test_window_audit],
        ignore_index=True,
    )
    if window_segment_audit["is_synthetic"].any():
        raise AssertionError("Real window audit unexpectedly contains synthetic data")

    windowing_summary = {
        "real_train_windows": real_train_windows,
        "synthetic_train_windows": synthetic_train_windows,
        "final_train_windows": len(y_train),
        "real_validation_windows": real_val_windows,
        "real_test_windows": real_test_windows,
        "validation_synthetic_windows": 0,
        "test_synthetic_windows": 0,
        "smote_generated_windows": generated_total,
        "enn_retained_windows": retained_total,
        "enn_removed_windows": removed_total,
    }
    logger.info(f"Windowing summary: {windowing_summary}")

    logger.info("\nFinal leakage-safe split windows:")
    logger.info(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"  X_val:   {X_val.shape} | y_val:   {y_val.shape}")
    logger.info(f"  X_test:  {X_test.shape} | y_test:  {y_test.shape}")
    logger.info(f"  Best window size: {engineer.best_window_size}")
    logger.info(f"  Best step size: {engineer.best_step_size}")
    logger.info(f"  Selected features: {len(selected_features)}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "selected_features": selected_features,
        "columns_to_drop": columns_to_drop,
        "scaler": scaler,
        "label_classes": label_classes,
        "window_segment_audit": window_segment_audit,
        "windowing_summary": windowing_summary,
        "smote_audit": smote_audit,
        "best_window_size": engineer.best_window_size,
        "best_step_size": engineer.best_step_size,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import logging

    # Fix import path
    sys.path.append(str(Path(__file__).parent.parent))

    from config.config import (
        APPLY_SMOTE_PRE_WINDOW,
        BASE_DIR,
        WINDOW_SIZES,
        STEP_SIZES,
        VALID_SIZE,
        TEST_SIZE,
        RF_N_ESTIMATORS,
        CV_SPLITS
    )

    # -------------------- LOGGING --------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("\n")
    logger.info("█" * 80)
    logger.info("█" + " " * 20 + "STEP 3: FEATURE ENGINEERING (STANDALONE)" + " " * 20 + "█")
    logger.info("█" * 80)

    # -------------------- LOAD STEP 2 OUTPUT --------------------
    data_path = BASE_DIR / "outputs" / "data"
    use_smote_pre_window = APPLY_SMOTE_PRE_WINDOW or _env_flag_enabled("USE_SMOTE_PRE_WINDOW")
    smote_input_file = data_path / "step_2_Processed_Data_smote.pkl"
    default_input_file = data_path / "step_2_Processed_Data.pkl"
    input_file = smote_input_file if use_smote_pre_window else default_input_file

    if not input_file.exists():
        logger.error("❌ Step 2 output not found: %s", input_file)
        if use_smote_pre_window:
            logger.error("Run SMOTE-ENN first: python scripts/smote_pre_window.py")
        else:
            logger.error("Run: python preprocessing/preprocessor.py first")
        exit(1)

    logger.info(f"📂 Loading processed data from: {input_file}")
    if use_smote_pre_window:
        logger.info("Using pre-window SMOTE-ENN processed data")
    df = pd.read_pickle(input_file)

    logger.info(f"✓ Loaded data shape: {df.shape}")

    # -------------------- FEATURE ENGINEERING --------------------
    try:
        result = engineer_features_no_leakage(
            df,
            WINDOW_SIZES,
            STEP_SIZES,
            target_column='Label',
            valid_size=VALID_SIZE,
            test_size=TEST_SIZE,
            rf_n_estimators=RF_N_ESTIMATORS,
            cv_splits=CV_SPLITS
        )

        # -------------------- SAVE OUTPUT --------------------
        output_path = BASE_DIR / "outputs" / "data"
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / "X_train.npy", result["X_train"])
        np.save(output_path / "X_val.npy", result["X_val"])
        np.save(output_path / "X_test.npy", result["X_test"])
        np.save(output_path / "y_train.npy", result["y_train"])
        np.save(output_path / "y_val.npy", result["y_val"])
        np.save(output_path / "y_test.npy", result["y_test"])

        # Save metadata
        pd.to_pickle(result["selected_features"], output_path / "selected_features.pkl")
        pd.to_pickle(result["columns_to_drop"], output_path / "dropped_features.pkl")
        pd.to_pickle(result["scaler"], output_path / "standard_scaler.pkl")
        pd.to_pickle(result["label_classes"], output_path / "label_classes.pkl")
        result["window_segment_audit"].to_csv(
            output_path / "window_segment_audit.csv", index=False
        )
        result["smote_audit"].to_csv(
            output_path / "smote_audit.csv", index=False
        )
        pd.to_pickle(
            result["windowing_summary"],
            output_path / "windowing_summary.pkl",
        )
        pd.to_pickle(
            {
                "best_window_size": result["best_window_size"],
                "best_step_size": result["best_step_size"],
            },
            output_path / "window_config.pkl"
        )

        logger.info("\n💾 Saved outputs:")
        logger.info(f" - X_train.npy, X_val.npy, X_test.npy")
        logger.info(f" - y_train.npy, y_val.npy, y_test.npy")
        logger.info(f" - selected_features.pkl, dropped_features.pkl")
        logger.info(f" - standard_scaler.pkl, label_classes.pkl, window_config.pkl")
        logger.info(
            f" - window_segment_audit.csv, smote_audit.csv, windowing_summary.pkl"
        )

        logger.info("\n✅ STEP 3 COMPLETE")

    except Exception as e:
        logger.exception(f"❌ Feature engineering failed: {e}")
        raise
