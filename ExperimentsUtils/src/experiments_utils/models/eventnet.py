"""EventNet (Seeuws et al. 2024, EPFL SzCORE reproduction): 19-ch unipolar raw µV input, no dropout (no MCD)."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from experiments_utils.adversarial import fgsm_batch, pgd_batch
from eventnet.utils import SeizureDataset, events_to_mask, load_model

WINDOW_SEC = 120
THRESH = 0.44
BATCH_SIZE = 16


@lru_cache(maxsize=2)
def _model(device):
    """Load weights once per process (the local runner reuses them across recordings)."""
    return load_model().to(device).eval()


class _CenterProb(torch.nn.Module):
    """Event-centre probability map (batch, samples); differentiable target for adversarial attacks."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0][:, 0]


def _window_size_sec(recording_duration: float) -> float:
    """Same window-length rule as eventnet.main (shorter windows for recordings under ~4 min)."""
    if recording_duration - 5 < WINDOW_SEC * 2:
        return (recording_duration - 5) // 20 * 10
    return WINDOW_SEC


def _forward(model, dataset, device) -> tuple[np.ndarray, np.ndarray]:
    """(n_windows, samples) centre and duration maps, attacking each batch first when ATTACK is set.

    Upstream runs one window per batch; the network has no cross-window state, so batching
    gives the same per-window outputs.
    """
    attack = os.getenv("ATTACK", "").lower()
    eps = float(os.getenv("EPS", "0") or 0)
    mode = os.getenv("ATTACK_MODE", "flip").lower()
    thresh = float(os.getenv("ATTACK_THRESH", "0.5"))
    target = _CenterProb(model).eval()

    centers, durations = [], []
    for start in range(0, len(dataset), BATCH_SIZE):
        x = torch.stack([dataset[i] for i in range(start, min(start + BATCH_SIZE, len(dataset)))]).to(device)
        if attack == "fgsm" and eps > 0:
            x = fgsm_batch(target, x, mode=mode, eps=eps, thresh=thresh)
        elif attack == "pgd" and eps > 0:
            alpha = float(os.getenv("ALPHA", "0.005"))
            steps = int(os.getenv("STEPS", "5"))
            x = pgd_batch(target, x, mode=mode, eps=eps, alpha=alpha, steps=steps, thresh=thresh)
        with torch.no_grad():
            c, d = model(x)
        centers.append(c[:, 0].cpu().numpy())
        durations.append(d[:, 0].cpu().numpy())
    return np.concatenate(centers), np.concatenate(durations)


def detect(eeg, data) -> list[int]:
    """EventNet: (possibly perturbed) unipolar µV array -> binary seizure mask (same as eventnet.main)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = np.zeros(data.shape[1], dtype=int)
    win_sec = _window_size_sec(data.shape[1] / eeg.fs)
    if win_sec <= 25:
        return mask.tolist()

    dataset = SeizureDataset(data=data, window_size_sec=win_sec, fs=eeg.fs)
    if len(dataset) == 0:
        return mask.tolist()
    centers, durations = _forward(_model(device), dataset, device)
    preds = np.concatenate(events_to_mask(centers, durations, win_sec, fs=eeg.fs, thresh=THRESH))
    n = min(len(preds), len(mask))
    mask[:n] = preds[:n]
    return mask.tolist()
