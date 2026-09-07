"""PyTorch Dataset wrapper for large NumPy window arrays."""

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyWindowDataset(Dataset):
    """Read large window arrays lazily and convert each sample to tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: {len(X)} != {len(y)}")
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = np.array(self.X[index], dtype=np.float32, copy=True)
        y = int(self.y[index])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
