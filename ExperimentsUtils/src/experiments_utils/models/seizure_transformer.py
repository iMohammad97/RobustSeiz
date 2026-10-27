from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from experiments_utils.adversarial_predict import forward_with_adversarial
from wu_2025.utils import (
    compute_agreement_trace,
    eventize_probabilities,
    forward_windows,
    get_dataloader,
    load_models,
)


@lru_cache(maxsize=2)
def _model(device):
    """Load weights once per process (the local runner reuses them across recordings)."""
    return load_models(device)


def detect(eeg, data) -> list[int]:
    """SeizureTransformer: perturbed EEG array -> binary seizure mask."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = eeg.data.shape[1]

    model = _model(device)
    dataloader = get_dataloader(data, eeg.fs, batch_size=512)

    if os.getenv("ATTACK", "").lower() in ("fgsm", "pgd") and float(os.getenv("EPS", "0") or 0) > 0:
        probs = forward_with_adversarial(model, dataloader, device)
    else:
        probs = forward_windows(model, dataloader, device)

    return eventize_probabilities(probs, seq_len).tolist()


def compute_agreement(eeg, data) -> np.ndarray:
    """SeizureTransformer: perturbed EEG -> per-sample MCD agreement index φ in [0, 1]."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = eeg.data.shape[1]
    n_samples = int(os.getenv("MCD_SAMPLES", "20"))
    thresh = float(os.getenv("MCD_THRESH", "0.8"))

    model = _model(device)
    dataloader = get_dataloader(data, eeg.fs, batch_size=512)
    return compute_agreement_trace(model, dataloader, device, seq_len, n_samples, thresh)
