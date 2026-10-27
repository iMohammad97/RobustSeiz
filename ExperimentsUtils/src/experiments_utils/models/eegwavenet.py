"""EEGWaveNet (Thuwajit et al. 2021, EPFL SzCORE reproduction): 18-ch bipolar input, no dropout (no MCD)."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from experiments_utils.adversarial import fgsm_batch, pgd_batch
from thuwajit_2021.utils import get_dataloader, load_models

WINDOW_SEC = 4
OVERLAP_SEC = 3


@lru_cache(maxsize=2)
def _models(device):
    """Five cross-validation fold models, loaded once per process."""
    return load_models(device)


class _EnsembleSeizureProb(torch.nn.Module):
    """Mean seizure probability over folds; differentiable target for adversarial attacks."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([m(x).exp()[:, 1] for m in self.models]).mean(0)


def _window_votes(models, dataloader, device) -> np.ndarray:
    """(n_models, n_windows) argmax labels, attacking each batch first when ATTACK is set."""
    attack = os.getenv("ATTACK", "").lower()
    eps = float(os.getenv("EPS", "0") or 0)
    mode = os.getenv("ATTACK_MODE", "flip").lower()
    thresh = float(os.getenv("ATTACK_THRESH", "0.5"))
    ensemble = _EnsembleSeizureProb(models).eval()

    votes = [[] for _ in models]
    for data in dataloader:
        data = data.float().to(device)
        if attack == "fgsm" and eps > 0:
            data = fgsm_batch(ensemble, data, mode=mode, eps=eps, thresh=thresh)
        elif attack == "pgd" and eps > 0:
            alpha = float(os.getenv("ALPHA", "0.005"))
            steps = int(os.getenv("STEPS", "5"))
            data = pgd_batch(ensemble, data, mode=mode, eps=eps, alpha=alpha, steps=steps, thresh=thresh)
        with torch.no_grad():
            for k, model in enumerate(models):
                votes[k].extend(model(data).argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(votes)


def _aggregate(votes: np.ndarray, recording_duration: float, fs: float) -> np.ndarray:
    """Same per-sample majority vote as thuwajit_2021.utils.predict."""
    window_size = int(WINDOW_SEC * fs)
    overlap_size = int(OVERLAP_SEC * fs)
    detections = np.zeros((len(votes), int(recording_duration * fs)))
    for k, y_preds in enumerate(votes):
        for i in range(len(y_preds)):
            left = max(0, i * (window_size - overlap_size))
            right = min(recording_duration * fs, (i + 1) * (window_size - overlap_size))
            majority = np.mean(
                y_preds[max(0, i - int(OVERLAP_SEC // 2)) : min(i + int(OVERLAP_SEC // 2), recording_duration)]
            ) > 0.5
            detections[k, int(left) : int(right)] = int(majority)
    return (np.mean(detections, axis=0) > 0.5).astype(int)


def detect(eeg, data) -> list[int]:
    """EEGWaveNet: (possibly perturbed) bipolar µV array -> binary seizure mask."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recording_duration = data.shape[1] / eeg.fs
    dataloader = get_dataloader(data, WINDOW_SEC, eeg.fs, batch_size=512)
    votes = _window_votes(_models(device), dataloader, device)
    return _aggregate(votes, recording_duration, eeg.fs).tolist()
