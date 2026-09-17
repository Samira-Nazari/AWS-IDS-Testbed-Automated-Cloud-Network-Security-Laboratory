"""
Standalone script to check Kaggle credentials
Run this before the main pipeline to verify setup
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_kaggle_credentials() -> bool:
    """
    Check if Kaggle credentials are configured
    
    Returns:
        True if credentials exist, False otherwise
    """
    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
    
    logger.info(f"Checking for Kaggle credentials at: {kaggle_config}")
    
    if kaggle_config.exists():
        logger.info("✓ Kaggle credentials found!")
        logger.info(f"  File: {kaggle_config}")
        logger.info(f"  Size: {kaggle_config.stat().st_size} bytes")
        return True
    else:
        logger.error("❌ Kaggle credentials not found!")
        logger.error(f"  Expected at: {kaggle_config}")
        logger.error("\n📋 To set up Kaggle credentials:")
        logger.error("   1. Go to https://www.kaggle.com/settings/account")
        logger.error("   2. Click 'Create New API Token'")
        logger.error("   3. Download and save kaggle.json to ~/.kaggle/")
        logger.error("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False


def check_kaggle_cli() -> bool:
    """
    Check if kaggle CLI is installed
    
    Returns:
        True if kaggle CLI is available, False otherwise
    """
    import subprocess
    
    logger.info("\nChecking Kaggle CLI installation...")
    
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ Kaggle CLI is installed")
            logger.info(f"  Version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    logger.error("❌ Kaggle CLI not found!")
    logger.error("   Install with: pip install kaggle")
    return False


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("KAGGLE CREDENTIALS CHECK")
    logger.info("="*60 + "\n")
    
    # Check credentials
    creds_ok = check_kaggle_credentials()
    
    # Check CLI
    cli_ok = check_kaggle_cli()
    
    # Summary
    logger.info("\n" + "="*60)
    if creds_ok and cli_ok:
        logger.info("✓ All checks passed! You're ready to run the pipeline.")
    else:
        logger.info("❌ Some checks failed. Please fix the issues above.")
    logger.info("="*60)
