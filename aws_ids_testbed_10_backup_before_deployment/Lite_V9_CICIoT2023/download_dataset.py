"""
Standalone script to download CIC-DDoS2019 dataset from Kaggle
Run this separately to verify data download before full pipeline
"""

import logging
import os
from pathlib import Path
from config.config import KAGGLE_DATASET_NAME, KAGGLE_DATASET_PATH
from data.data_loader import KaggleDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def download_dataset():
    """
    Download CIC-DDoS2019 dataset from Kaggle
    """
    logger.info("="*60)
    logger.info("KAGGLE DATASET DOWNLOAD")
    logger.info("="*60 + "\n")
    
    logger.info(f"Dataset: {KAGGLE_DATASET_NAME}")
    logger.info(f"Download Path: {KAGGLE_DATASET_PATH}\n")
    
    # Create data loader
    loader = KaggleDataLoader(KAGGLE_DATASET_NAME, KAGGLE_DATASET_PATH)
    
    # Check credentials first
    logger.info("Step 1: Verifying Kaggle credentials...")
    if not loader.check_kaggle_credentials():
        logger.error("\n❌ Kaggle credentials check failed!")
        logger.error("   Please set up your Kaggle API token first.")
        logger.error("   Run: python check_kaggle_credentials.py")
        return False
    
    logger.info("✓ Kaggle credentials verified\n")
    
    # Download dataset
    logger.info("Step 2: Downloading dataset from Kaggle...")
    logger.info("   This may take a while depending on dataset size and internet speed...\n")
    
    success = loader.download_dataset()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("✓ Dataset downloaded successfully!")
        logger.info("="*60)
        
        # Check what was downloaded
        files_in_dir = []
        if os.path.exists(KAGGLE_DATASET_PATH):
            files_in_dir = os.listdir(KAGGLE_DATASET_PATH)
            logger.info(f"\nFiles in {KAGGLE_DATASET_PATH}:")
            for f in files_in_dir:
                file_path = os.path.join(KAGGLE_DATASET_PATH, f)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024*1024)
                    logger.info(f"  - {f} ({size_mb:.2f} MB)")
                else:
                    logger.info(f"  - {f}/ (directory)")
        
        return True
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ Dataset download failed!")
        logger.error("="*60)
        return False


if __name__ == "__main__":
    success = download_dataset()
    exit(0 if success else 1)
