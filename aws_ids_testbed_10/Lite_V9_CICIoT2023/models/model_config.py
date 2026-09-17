"""
Public model factory for Lite_V9_CICIoT2023.

This file keeps the existing create_model(...) API used by main.py,
training/trainer.py, and evaluation/evaluator.py, but routes model creation to
the Lite Transformer encoder classifier.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from models.lite_encoder_classifier import (
        LiteTransformerConfig,
        LiteTransformerEncoderClassifier,
        create_lite_transformer_classifier,
    )
except ModuleNotFoundError as exc:
    if exc.name != "models":
        raise
    from lite_encoder_classifier import (
        LiteTransformerConfig,
        LiteTransformerEncoderClassifier,
        create_lite_transformer_classifier,
    )

logger = logging.getLogger(__name__)

create_model = create_lite_transformer_classifier

__all__ = [
    "LiteTransformerConfig",
    "LiteTransformerEncoderClassifier",
    "create_lite_transformer_classifier",
    "create_model",
]


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    sys.path.insert(0, str(BASE_DIR))

    from config.config import (
        LITE_D_MODEL,
        LITE_N_HEADS,
        LITE_D_FF,
        LITE_DROPOUT,
        LITE_ACTIVATION,
        LITE_N_LAYERS,
        LITE_FC_DROPOUT,
        LITE_USE_SHORT_CONV_BRANCH,
        LITE_CONV_KERNEL_SIZE,
        LITE_CONV_NUM_LAYERS,
        LITE_FUSION_TYPE,
        LITE_ATTENTION_FRACTION,
        LITE_SANITY_C_IN,
        LITE_SANITY_SEQ_LEN,
        LITE_SANITY_C_OUT,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

    data_path = BASE_DIR / "outputs" / "data"
    model_config_path = BASE_DIR / "outputs" / "models" / "model_config.json"

    saved_model_config = {}
    if model_config_path.exists():
        with open(model_config_path, "r") as f:
            saved_model_config = json.load(f)

    x_train_path = data_path / "X_train.npy"
    y_train_path = data_path / "y_train.npy"

    if x_train_path.exists():
        x_train = np.load(x_train_path, mmap_mode="r")
        c_in = x_train.shape[2]
        seq_len = x_train.shape[1]
    elif saved_model_config:
        c_in = saved_model_config["c_in"]
        seq_len = saved_model_config["seq_len"]
    else:
        logger.warning(
            "X_train.npy/model_config.json not found; using Lite sanity dimensions "
            "from config.py."
        )
        c_in = LITE_SANITY_C_IN
        seq_len = LITE_SANITY_SEQ_LEN

    if y_train_path.exists():
        y_train = np.load(y_train_path, mmap_mode="r")
        c_out = len(np.unique(y_train))
    elif saved_model_config:
        c_out = saved_model_config["c_out"]
    else:
        logger.warning(
            "y_train.npy/model_config.json not found; using Lite sanity class count "
            "from config.py."
        )
        c_out = LITE_SANITY_C_OUT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = create_model(
        c_in=c_in,
        c_out=c_out,
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

    dummy_input = torch.randn(4, seq_len, c_in, device=device)
    output = model(dummy_input)

    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    assert output.shape == (4, c_out)
