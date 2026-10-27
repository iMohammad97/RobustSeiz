"""EEGNet (braindecode) trained by us on TUSZ train+dev: 18-ch bipolar input, dropout -> MCD."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from eegnet_seizure.model import load
from eegnet_seizure.predict import agreement, dataloader, eventize, per_second, to_samples, window_probs
from eegnet_seizure.preprocessing import preprocess
from experiments_utils.adversarial_predict import forward_with_adversarial


@lru_cache(maxsize=2)
def _model(device):
    """Weights and eventization config, loaded once per process."""
    return load(device)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect(eeg, data) -> list[int]:
    """EEGNet: (possibly perturbed) bipolar µV array -> binary seizure mask."""
    device = _device()
    model, config = _model(device)
    x = preprocess(data, eeg.fs)
    loader, starts = dataloader(x)
    if os.getenv("ATTACK", "").lower() in ("fgsm", "pgd") and float(os.getenv("EPS", "0") or 0) > 0:
        probs = forward_with_adversarial(model, loader, device).detach().cpu().numpy()
    else:
        probs = window_probs(model, loader, device)
    sec = per_second(probs, starts, x.shape[1])
    decisions = eventize(sec, config["threshold"], config["close_sec"], config["min_event_sec"])
    return to_samples(decisions, data.shape[1], eeg.fs).tolist()


def compute_agreement(eeg, data) -> np.ndarray:
    """EEGNet: per-sample MCD agreement index φ in [0, 1] (T dropout passes)."""
    device = _device()
    model, _ = _model(device)
    x = preprocess(data, eeg.fs)
    phi = agreement(
        model,
        x,
        device,
        n_samples=int(os.getenv("MCD_SAMPLES", "20")),
        thresh=float(os.getenv("MCD_THRESH", "0.8")),
    )
    return to_samples(phi, data.shape[1], eeg.fs)
