"""Model-agnostic helpers for Monte Carlo dropout uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.nn as nn


def enable_dropout_only(model: nn.Module) -> None:
    """Keep BatchNorm (and other non-dropout layers) in eval mode; enable dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def compute_agreement_index(binary_runs: np.ndarray) -> np.ndarray:
    """
    Agreement index φ across Monte Carlo dropout runs.

    Args:
        binary_runs: shape (T, N) with values in {0, 1}

    Returns:
        Per-sample agreement in [0, 1], shape (N,).
    """
    if binary_runs.ndim != 2:
        raise ValueError(f"binary_runs must be 2-D (T, N); got shape {binary_runs.shape}")
    return binary_runs.mean(axis=0).astype(np.float32)


def agreement_output_path(events_path: str | Path) -> Path:
    """Map an events TSV path to the matching agreement NPZ path."""
    path = Path(events_path)
    stem = path.name
    if stem.endswith("_events.tsv"):
        stem = stem[: -len("_events.tsv")] + "_agreement.npz"
    elif stem.endswith(".tsv"):
        stem = stem[: -4] + "_agreement.npz"
    else:
        stem = path.stem + "_agreement.npz"
    return path.with_name(stem)
