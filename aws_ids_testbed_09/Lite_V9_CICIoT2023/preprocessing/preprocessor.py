"""
Data preprocessing module for the V19 training pipeline.
Handles cleaning, optional standardization, and protected metadata columns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

SPECIAL_COLUMNS = (
    "Label",
    "Timestamp",
    "unique_id",
    "planned_split",
    "split_segment_id",
    "source_row_index",
)


class DataPreprocessor:
    """Preprocess raw data for model training"""
    
    def __init__(self, columns_to_drop: list = None):
        """
        Initialize preprocessor
        
        Args:
            columns_to_drop: List of column names to drop
        """
        self.columns_to_drop = columns_to_drop or []
        self.scaler = None
        self.feature_columns = None

    def sort_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort data chronologically by Timestamp
    
        Args:
            df: Input DataFrame
        
        Returns:
            Sorted DataFrame
        """
        if 'Timestamp' in df.columns:
            logger.info("\n🕒 Sorting data by Timestamp...")
            df = df.sort_values(by='Timestamp').reset_index(drop=True)
            logger.info("   ✓ Data sorted chronologically")
        else:
            logger.warning("⚠️ Timestamp column not found — skipping sorting")
        return df
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and infinite values
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DATA PREPROCESSING & CLEANING")
        logger.info("="*60)
        
        logger.info("\n📋 Data shape before cleaning:")
        logger.info(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Temporarily remove protected metadata columns from numeric cleaning.
        protected_columns = {
            col: df[col].copy()
            for col in SPECIAL_COLUMNS
            if col in df.columns
        }
        df_features = df.drop(columns=list(protected_columns), errors="ignore")
        
        logger.info("\n🧹 Checking for infinite and missing values...")
        
        # Select numeric columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        numeric_df = df_features[numeric_cols]
        
        # Check for infinite values before cleaning
        inf_count_before = np.isinf(numeric_df).sum().sum()
        logger.info(f"   Infinite values before cleaning: {inf_count_before}")
        
        # Replace infinite values with 0
        df_features[numeric_cols] = df_features[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        
        # Fill NaN values with 0
        df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
        
        # Verify no infinite values remain
        numeric_df = df_features[numeric_cols]
        inf_count_after = np.isinf(numeric_df).sum().sum()
        logger.info(f"   Infinite values after cleaning: {inf_count_after}")
        
        # Check for missing values
        missing_count = df_features.isnull().sum().sum()
        logger.info(f"   Missing values: {missing_count}")
        
        # Restore protected metadata columns.
        for col, values in protected_columns.items():
            df_features[col] = values
        
        logger.info(f"\n✓ Data shape after cleaning:")
        logger.info(f"   Rows: {df_features.shape[0]}, Columns: {df_features.shape[1]}")
        
        return df_features
    
    def drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop irrelevant or redundant columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with dropped columns
        """
        logger.info("\n📌 Dropping irrelevant columns...")
        logger.info(f"   Initial columns: {df.shape[1]}")
        
        protected_drop_columns = set(SPECIAL_COLUMNS)
        requested_protected = [
            col for col in self.columns_to_drop
            if col in protected_drop_columns and col in df.columns
        ]
        if requested_protected:
            logger.info(
                "   Preserving protected columns: "
                f"{', '.join(requested_protected)}"
            )

        # Only drop columns that exist and are not required metadata.
        cols_to_drop = [
            col for col in self.columns_to_drop
            if col in df.columns and col not in protected_drop_columns
        ]
        
        if cols_to_drop:
            logger.info(f"   Columns to drop: {len(cols_to_drop)}")
            for col in cols_to_drop[:5]:  # Show first 5
                logger.info(f"     - {col}")
            if len(cols_to_drop) > 5:
                logger.info(f"     ... and {len(cols_to_drop) - 5} more")
            
            df = df.drop(columns=cols_to_drop)
        
        logger.info(f"   Final columns: {df.shape[1]}")
        
        return df
    
    def standardize_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Standardize features (zero mean, unit variance)
        
        Args:
            df: Input DataFrame
            fit_scaler: If True, fit scaler on this data. If False, use existing scaler.
            
        Returns:
            Standardized DataFrame
        """
        logger.info("\n⚖️  Standardizing features (zero mean, unit variance)...")
        
        # Make a copy
        df = df.copy()
        
        # Identify feature columns and exclude protected metadata.
        exclude_cols = set(SPECIAL_COLUMNS)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        logger.info(f"   Features to standardize: {len(feature_cols)}")
        
        # Select numeric columns only
        numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"   Numeric features: {len(numeric_feature_cols)}")
        
        # Fit or use existing scaler
        if fit_scaler:
            logger.info("   Fitting scaler on training data...")
            self.scaler = StandardScaler()
            standardized_data = self.scaler.fit_transform(df[numeric_feature_cols])
        else:
            if self.scaler is None:
                raise ValueError("❌ Scaler not fitted. Use fit_scaler=True first.")
            logger.info("   Using existing scaler...")
            standardized_data = self.scaler.transform(df[numeric_feature_cols])
        
        # Create new dataframe with standardized data
        standardized_df = pd.DataFrame(standardized_data, columns=numeric_feature_cols, index=df.index)
        
        # Add back non-numeric and special columns
        for col in exclude_cols:
            if col in df.columns:
                standardized_df[col] = df[col].values
        
        logger.info(f"   ✓ Standardization complete")
        logger.info(f"     - Mean of features: {standardized_df[numeric_feature_cols].mean().mean():.6f}")
        logger.info(f"     - Std of features: {standardized_df[numeric_feature_cols].std().mean():.6f}")
        
        return standardized_df
    
    def preprocess(self, df: pd.DataFrame, fit_scaler: bool = True,
                   standardize: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            fit_scaler: If True, fit scaler. If False, use existing scaler.
            standardize: If True, standardize features. Set False when the
                train/validation/test split must happen before scaling.
            
        Returns:
            Fully preprocessed DataFrame
        """
        # STEP 0: SORT FIRST (VERY IMPORTANT)
        df = self.sort_by_timestamp(df)
        
        # Step 1: Clean data
        df = self.clean_data(df)
        
        # Step 2: Drop irrelevant columns
        df = self.drop_irrelevant_columns(df)
        
        # Step 3: Standardize features
        if standardize:
            df = self.standardize_features(df, fit_scaler=fit_scaler)
        else:
            logger.info("\n⚖️  Skipping standardization until after data split")
        
        logger.info("\n✓ Preprocessing pipeline complete!")
        logger.info("="*60 + "\n")
        
        return df


def preprocess_data(df: pd.DataFrame, columns_to_drop: list = None,
                   fit_scaler: bool = True,
                   standardize: bool = True) -> pd.DataFrame:
    """
    Main function for preprocessing
    
    Args:
        df: Input DataFrame
        columns_to_drop: Columns to drop
        fit_scaler: Whether to fit the scaler
        standardize: Whether to standardize features during preprocessing
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor(columns_to_drop=columns_to_drop)
    return preprocessor.preprocess(df, fit_scaler=fit_scaler,
                                   standardize=standardize)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import logging

    # Fix import path
    sys.path.append(str(Path(__file__).parent.parent))

    from config.config import COLUMNS_TO_DROP

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("\n")
    logger.info("█" * 80)
    logger.info("█" + " " * 20 + "STEP 2: PREPROCESS DATA (STANDALONE)" + " " * 20 + "█")
    logger.info("█" * 80)

    # -------------------- LOAD STEP 1 OUTPUT --------------------
    # output_path = Path("outputs/data")
    from config.config import BASE_DIR

    output_path = BASE_DIR / "outputs" / "data"
    input_path = Path(output_path / "step_1_Loaded_Data.pkl")

    if not input_path.exists():
        logger.error("❌ Step 1 output not found!")
        logger.error("Run: python data/data_loader.py first")
        exit(1)

    logger.info(f"📂 Loading raw data from: {input_path}")
    df = pd.read_pickle(input_path)

    logger.info(f"✓ Loaded data shape: {df.shape}")

    # -------------------- PREPROCESS --------------------
    preprocessor = DataPreprocessor(columns_to_drop=COLUMNS_TO_DROP)

    processed_df = preprocessor.preprocess(df, fit_scaler=True,
                                           standardize=False)

    # -------------------- SAVE OUTPUT --------------------
    #output_path = Path("outputs/data")
    from config.config import BASE_DIR
    output_path = BASE_DIR / "outputs" / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    # save_file = output_path / "step_2_Processed_Data.pkl"
    save_file = output_path / "step_2_Processed_Data.pkl"
    processed_df.to_pickle(save_file)

    logger.info(f"\n💾 Saved processed data to: {save_file}")

    logger.info("\n✅ STEP 2 COMPLETE")
