"""Windowed inference, per-second aggregation and eventization."""

from __future__ import annotations

import math

import numpy as np
import torch

from eegnet_seizure.preprocessing import MODEL_FS, WINDOW, window_starts

INFER_STRIDE = MODEL_FS  # 1 s


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, starts: np.ndarray):
        self.x = x
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return torch.from_numpy(np.ascontiguousarray(self.x[:, s : s + WINDOW]))


def dataloader(x: np.ndarray, batch_size: int = 2048):
    starts = window_starts(x.shape[1], INFER_STRIDE)
    loader = torch.utils.data.DataLoader(WindowDataset(x, starts), batch_size=batch_size, shuffle=False)
    return loader, starts


def window_probs(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append(model(batch.float().to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def per_second(probs: np.ndarray, starts: np.ndarray, n_samples: int) -> np.ndarray:
    """Average every window probability over the seconds it covers."""
    n_sec = math.ceil(n_samples / MODEL_FS)
    total = np.zeros(n_sec)
    count = np.zeros(n_sec)
    for p, s in zip(probs, starts):
        a, b = s // MODEL_FS, math.ceil((s + WINDOW) / MODEL_FS)
        total[a:b] += p
        count[a:b] += 1
    return np.divide(total, count, out=np.zeros(n_sec), where=count > 0)


def eventize(sec_probs: np.ndarray, thresh: float, close_sec: int = 2, min_event_sec: int = 2) -> np.ndarray:
    """Per-second probabilities -> per-second binary decisions (fill short gaps, drop short events)."""
    y = (sec_probs >= thresh).astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], y, [0]))))
    events = list(zip(edges[::2], edges[1::2]))
    merged = []
    for a, b in events:
        if merged and a - merged[-1][1] <= close_sec:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = np.zeros_like(y)
    for a, b in merged:
        if b - a >= min_event_sec:
            out[a:b] = 1
    return out


def to_samples(per_sec: np.ndarray, n_samples: int, fs: float) -> np.ndarray:
    """Per-second values -> per-sample array at the recording's sampling rate."""
    idx = np.minimum((np.arange(n_samples) / fs).astype(int), len(per_sec) - 1)
    return per_sec[idx]


def enable_dropout(model) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def agreement(model, x: np.ndarray, device, n_samples: int, thresh: float) -> np.ndarray:
    """MCD agreement index per second: fraction of dropout passes voting seizure."""
    loader, starts = dataloader(x)
    enable_dropout(model)
    votes = []
    with torch.no_grad():
        for _ in range(n_samples):
            probs = np.concatenate([model(b.float().to(device)).cpu().numpy() for b in loader])
            votes.append(per_second(probs, starts, x.shape[1]) >= thresh)
    model.eval()
    return np.mean(votes, axis=0)
