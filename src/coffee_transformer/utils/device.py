"""Device selection. Target is CUDA (Section 8); falls back to CPU for the
local debug pass the design mandates before renting GPU hours."""

from __future__ import annotations

import torch


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
