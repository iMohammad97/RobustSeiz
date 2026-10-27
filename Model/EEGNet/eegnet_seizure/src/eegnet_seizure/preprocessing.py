"""Signal preprocessing shared by training and inference."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

MODEL_FS = 128
WINDOW_SEC = 4
WINDOW = MODEL_FS * WINDOW_SEC
BAND_HZ = (0.5, 40.0)


def preprocess(data: np.ndarray, fs: float) -> np.ndarray:
    """(C, T) bipolar µV at ``fs`` -> (C, T') float32 at 128 Hz, band-passed and z-scored per channel."""
    sos = butter(4, BAND_HZ, btype="bandpass", fs=fs, output="sos")
    out = []
    for channel in np.asarray(data, dtype=np.float64):
        y = sosfiltfilt(sos, channel)
        if int(fs) != MODEL_FS:
            y = resample_poly(y, MODEL_FS, int(fs))
        sd = y.std()
        out.append(((y - y.mean()) / (sd if sd > 1e-6 else 1.0)).astype(np.float32))
    return np.stack(out)


def window_starts(n_samples: int, stride: int) -> np.ndarray:
    """Start indices of full windows; the last window is aligned to the recording end."""
    if n_samples < WINDOW:
        return np.zeros(0, dtype=int)
    starts = np.arange(0, n_samples - WINDOW + 1, stride)
    if starts[-1] != n_samples - WINDOW:
        starts = np.append(starts, n_samples - WINDOW)
    return starts
