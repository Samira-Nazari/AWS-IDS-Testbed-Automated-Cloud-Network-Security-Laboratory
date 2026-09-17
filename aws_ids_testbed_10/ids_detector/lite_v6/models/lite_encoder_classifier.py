"""
Lite Transformer encoder classifier for CIC-DDoS2019 sliding windows.

Input shape:
    (batch, seq_len, c_in)

Output shape:
    (batch, c_out)
"""

from dataclasses import asdict, dataclass
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _resolve_long_dim(d_model: int, n_heads: int, attention_fraction: float) -> int:
    if d_model <= 1:
        raise ValueError("d_model must be greater than 1.")
    if n_heads < 1:
        raise ValueError("n_heads must be at least 1.")
    if not 0.0 < attention_fraction < 1.0:
        raise ValueError("attention_fraction must be between 0 and 1.")

    long_dim = int(round(d_model * attention_fraction))
    long_dim = max(n_heads, long_dim)
    long_dim = long_dim - (long_dim % n_heads)

    if long_dim <= 0:
        long_dim = n_heads
    if long_dim >= d_model:
        raise ValueError(
            "Lite split leaves no short-range channels. "
            "Reduce attention_fraction or increase d_model."
        )
    if long_dim % n_heads != 0:
        raise ValueError("Long-range attention dimension must divide evenly by n_heads.")

    return long_dim


class FeatureEmbedding(nn.Module):
    """Project per-flow numeric features into the Lite Transformer model space."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ShortRangeConvBranch(nn.Module):
    """Depthwise temporal convolution branch for local short-range patterns."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dropout: float,
        num_conv_layers: int,
        activation: str,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve sequence length.")
        if num_conv_layers < 1:
            raise ValueError("num_conv_layers must be at least 1.")

        self.gate = nn.Linear(dim, 2 * dim)
        self.glu = nn.GLU(dim=-1)

        layers = []
        for layer_idx in range(num_conv_layers):
            current_kernel = kernel_size if layer_idx == 0 else 3
            padding = current_kernel // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=dim,
                        out_channels=dim,
                        kernel_size=current_kernel,
                        padding=padding,
                        groups=dim,
                    ),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.Dropout(dropout),
                ]
            )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.glu(self.gate(x))
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x + residual


class LiteEncoderLayer(nn.Module):
    """
    Lite Transformer encoder layer.

    The hidden channels are split into:
      - long-range branch: multi-head self-attention
      - short-range branch: lightweight/depthwise temporal convolution
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation: str,
        use_short_conv_branch: bool,
        conv_kernel_size: int,
        conv_num_layers: int,
        attention_fraction: float,
    ):
        super().__init__()
        self.use_short_conv_branch = use_short_conv_branch

        if use_short_conv_branch:
            self.long_dim = _resolve_long_dim(d_model, n_heads, attention_fraction)
            self.short_dim = d_model - self.long_dim
        else:
            if d_model % n_heads != 0:
                raise ValueError("d_model must divide evenly by n_heads when using full attention.")
            self.long_dim = d_model
            self.short_dim = 0

        self.long_attention = nn.MultiheadAttention(
            embed_dim=self.long_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        if self.use_short_conv_branch:
            self.short_conv = ShortRangeConvBranch(
                dim=self.short_dim,
                kernel_size=conv_kernel_size,
                dropout=dropout,
                num_conv_layers=conv_num_layers,
                activation=activation,
            )

        self.branch_projection = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.use_short_conv_branch:
            long_x = x[..., : self.long_dim]
            short_x = x[..., self.long_dim :]
            long_out, _ = self.long_attention(long_x, long_x, long_x, need_weights=False)
            short_out = self.short_conv(short_x)
            x = torch.cat([long_out, short_out], dim=-1)
        else:
            x, _ = self.long_attention(x, x, x, need_weights=False)

        x = self.branch_projection(x)
        x = self.norm1(residual + self.dropout(x))

        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout(x))
        return x


class LiteTransformerEncoderClassifier(nn.Module):
    """Encoder-only Lite Transformer classifier for time-series windows."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        n_layers: int = 3,
        fc_dropout: float = 0.1,
        use_short_conv_branch: bool = True,
        conv_kernel_size: int = 5,
        conv_num_layers: int = 2,
        attention_fraction: float = 0.5,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.d_model = d_model

        self.embedding = FeatureEmbedding(c_in, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                LiteEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    use_short_conv_branch=use_short_conv_branch,
                    conv_kernel_size=conv_kernel_size,
                    conv_num_layers=conv_num_layers,
                    attention_fraction=attention_fraction,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            _activation(activation),
            nn.Dropout(fc_dropout),
            nn.Linear(d_model, c_out),
        )

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (batch, seq_len, c_in), got {tuple(x.shape)}")
        if x.size(1) > self.seq_len:
            raise ValueError(
                f"Input seq_len {x.size(1)} is longer than configured seq_len {self.seq_len}."
            )

        x = self.embedding(x)
        x = x + self.position_embedding[:, : x.size(1), :]
        x = self.input_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


@dataclass
class LiteTransformerConfig:
    model_name: str
    c_in: int
    c_out: int
    seq_len: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    activation: str
    n_layers: int
    fc_dropout: float
    use_cnn_local_branch: bool
    conv_kernel_size: int
    cnn_num_conv_layers: int
    fusion_type: str
    attention_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def create_lite_transformer_classifier(
    c_in: int,
    c_out: int,
    seq_len: int,
    d_model: int = 64,
    n_heads: int = 4,
    d_ff: int = 256,
    dropout: float = 0.1,
    activation: str = "gelu",
    n_layers: int = 3,
    fc_dropout: float = 0.1,
    use_cnn_local_branch: bool = True,
    conv_kernel_size: int = 5,
    cnn_num_conv_layers: int = 2,
    fusion_type: str = "concat",
    attention_fraction: float = 0.5,
    device: str = "cpu",
) -> tuple[nn.Module, LiteTransformerConfig]:
    if fusion_type != "concat":
            raise ValueError("Lite_V6 currently supports only fusion_type='concat'.")

    config = LiteTransformerConfig(
        model_name="LiteTransformerEncoderClassifier",
        c_in=c_in,
        c_out=c_out,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        n_layers=n_layers,
        fc_dropout=fc_dropout,
        use_cnn_local_branch=use_cnn_local_branch,
        conv_kernel_size=conv_kernel_size,
        cnn_num_conv_layers=cnn_num_conv_layers,
        fusion_type=fusion_type,
        attention_fraction=attention_fraction,
    )

    model = LiteTransformerEncoderClassifier(
        c_in=c_in,
        c_out=c_out,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        n_layers=n_layers,
        fc_dropout=fc_dropout,
        use_short_conv_branch=use_cnn_local_branch,
        conv_kernel_size=conv_kernel_size,
        conv_num_layers=cnn_num_conv_layers,
        attention_fraction=attention_fraction,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Lite Transformer model initialized")
    logger.info(f"  Input features: {c_in}")
    logger.info(f"  Output classes: {c_out}")
    logger.info(f"  Sequence length: {seq_len}")
    logger.info(f"  d_model: {d_model}")
    logger.info(f"  n_heads: {n_heads}")
    logger.info(f"  d_ff: {d_ff}")
    logger.info(f"  n_layers: {n_layers}")
    logger.info(f"  Short-range conv branch: {use_cnn_local_branch}")
    logger.info(f"  Conv kernel size: {conv_kernel_size}")
    logger.info(f"  Conv layers: {cnn_num_conv_layers}")
    logger.info(f"  Attention fraction: {attention_fraction}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Device: {device}")

    return model, config


create_model = create_lite_transformer_classifier


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = create_lite_transformer_classifier(
        c_in=48,
        c_out=12,
        seq_len=30,
        d_model=48,
        n_heads=4,
        d_ff=256,
        n_layers=3,
        device=device,
    )
    dummy = torch.randn(4, 30, 48, device=device)
    output = model(dummy)
    print("dummy input:", tuple(dummy.shape))
    print("dummy output:", tuple(output.shape))
