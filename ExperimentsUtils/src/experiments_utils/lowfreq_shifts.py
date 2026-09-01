"""Low-frequency impedance and baseline drift shift implementations."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

# Band keys for impedance noise and drift sin_band mode.
# 0_1 = low-pass 1 Hz; others = band-pass on generated noise/drift.
LOWFREQ_BANDS: dict[str, tuple[str, float, float | None]] = {
    "0_1": ("lowpass", 0.0, 1.0),
    "0.05_0.5": ("bandpass", 0.05, 0.5),
    "0.1_1": ("bandpass", 0.1, 1.0),
}

IMPEDANCE_AMP_UV_PRESETS = (25.0, 50.0, 100.0, 200.0)


def _get_rng(seed):
    if seed in ("", None):
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _slug(s: str) -> str:
    return str(s).replace(".", "p").replace("-", "_")


def parse_lowfreq_band(band: str) -> tuple[str, float, float | None]:
    key = band.strip().lower().replace("-", "_")
    if key not in LOWFREQ_BANDS:
        raise ValueError(f"Unknown low-frequency band {band!r}; expected one of {tuple(LOWFREQ_BANDS)}")
    return LOWFREQ_BANDS[key]


def _filter_signal(x, fs, ftype, flo, fhi, order=2):
    """IIR filter along time axis; x shape (C, T)."""
    y = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    if ftype == "lowpass":
        cutoff = float(fhi)
        if cutoff <= 0 or cutoff >= nyq:
            raise ValueError(f"Low-pass cutoff {cutoff} must be in (0, Nyquist={nyq:.2f})")
        b, a = butter(order, cutoff / nyq, btype="low")
    else:
        if flo <= 0 or fhi <= flo or fhi >= nyq:
            raise ValueError(f"Band-pass ({flo}, {fhi}) invalid for Nyquist={nyq:.2f}")
        b, a = butter(order, [flo / nyq, fhi / nyq], btype="band")
    return lfilter(b, a, y, axis=1)


def _channel_mask(n_ch: int, mode: str, n_select: int, rng) -> np.ndarray:
    mode = mode.lower()
    if mode == "all":
        return np.ones(n_ch, dtype=bool)
    if mode == "one":
        mask = np.zeros(n_ch, dtype=bool)
        mask[int(rng.integers(0, n_ch))] = True
        return mask
    if mode == "subset":
        k = max(1, min(int(n_select), n_ch))
        mask = np.zeros(n_ch, dtype=bool)
        mask[rng.choice(n_ch, size=k, replace=False)] = True
        return mask
    raise ValueError(f"Unknown channel mode {mode!r}; expected all, one, or subset")


def _intermittent_time_mask(n_samples: int, duty: float, rng) -> np.ndarray:
    duty = float(np.clip(duty, 0.05, 0.95))
    mask = np.zeros(n_samples, dtype=bool)
    target = int(n_samples * duty)
    block = max(1, n_samples // 200)
    guard = 0
    while mask.sum() < target and guard < 10_000:
        start = int(rng.integers(0, max(1, n_samples - 1)))
        length = int(rng.integers(block, max(block + 1, n_samples // 50)))
        mask[start : min(n_samples, start + length)] = True
        guard += 1
    return mask


def _scale_artifact(x_ref, artifact, amp_mode, amp_ratio, amp_uv):
    """Scale artifact per channel; amp_mode ratio uses channel std of x_ref."""
    art = np.asarray(artifact, dtype=np.float64)
    if amp_mode == "uv":
        peak = np.max(np.abs(art), axis=1, keepdims=True) + 1e-12
        return art / peak * float(amp_uv)
    ref_std = np.std(np.asarray(x_ref, dtype=np.float64), axis=1, keepdims=True) + 1e-12
    art_std = np.std(art, axis=1, keepdims=True) + 1e-12
    return art / art_std * (float(amp_ratio) * ref_std)


def _apply_spatial_temporal_mask(artifact, ch_mask, time_mask=None):
    out = np.zeros_like(artifact)
    for i in range(artifact.shape[0]):
        if not ch_mask[i]:
            continue
        row = artifact[i]
        if time_mask is not None:
            row = row * time_mask
        out[i] = row
    return out


def apply_impedance_shift(
    x,
    fs,
    band="0_1",
    amp_mode="ratio",
    amp_ratio=0.25,
    amp_uv=50.0,
    channel_mode="all",
    n_channels=1,
    filter_order=2,
    rng=None,
):
    """
    B1 — impedance / poor-contact noise: white noise band-limited and added to channels.

    Models electrode impedance fluctuations and slow contact noise (not broadband AWGN).
    """
    rng = _get_rng(rng)
    x = np.asarray(x, dtype=np.float64)
    n_ch, n_t = x.shape
    ftype, flo, fhi = parse_lowfreq_band(band)
    white = rng.standard_normal((n_ch, n_t))
    slow = _filter_signal(white, fs, ftype, flo, fhi, order=filter_order)
    scaled = _scale_artifact(x, slow, amp_mode, amp_ratio, amp_uv)
    ch_mask = _channel_mask(n_ch, channel_mode, n_channels, rng)
    delta = _apply_spatial_temporal_mask(scaled, ch_mask)
    return (x + delta).astype(x.dtype, copy=False)


def _drift_sin(x, fs, freq, rng):
    n_ch, n_t = x.shape
    t = np.arange(n_t) / fs
    phases = rng.uniform(0, 2 * np.pi, size=n_ch)
    return np.array([np.sin(2 * np.pi * freq * t + phases[i]) for i in range(n_ch)])


def _drift_sin_band(x, fs, band, rng):
    ftype, flo, fhi = parse_lowfreq_band(band)
    if ftype != "bandpass" or fhi is None:
        raise ValueError(f"sin_band drift requires a band-pass key, got {band!r}")
    n_ch, n_t = x.shape
    t = np.arange(n_t) / fs
    freqs = rng.uniform(flo, fhi, size=n_ch)
    phases = rng.uniform(0, 2 * np.pi, size=n_ch)
    return np.array([np.sin(2 * np.pi * freqs[i] * t + phases[i]) for i in range(n_ch)])


def _drift_random_walk(x, fs, band, rng, filter_order=2):
    n_ch, n_t = x.shape
    steps = rng.standard_normal((n_ch, n_t))
    ftype, flo, fhi = parse_lowfreq_band(band)
    steps = _filter_signal(steps, fs, ftype, flo, fhi, order=filter_order)
    walk = np.cumsum(steps, axis=1)
    walk -= np.mean(walk, axis=1, keepdims=True)
    return walk


def _drift_piecewise(x, fs, rng, n_segments=8):
    n_ch, n_t = x.shape
    out = np.zeros((n_ch, n_t), dtype=np.float64)
    n_segments = max(2, int(n_segments))
    knots = np.sort(rng.choice(n_t, size=min(n_segments, n_t), replace=False))
    knots = np.unique(np.concatenate(([0], knots, [n_t - 1])))
    t = np.arange(n_t)
    for i in range(n_ch):
        values = rng.standard_normal(len(knots))
        out[i] = np.interp(t, knots, values)
    out -= np.mean(out, axis=1, keepdims=True)
    return out


def apply_drift_shift(
    x,
    fs,
    mode="sin",
    drift_hz=0.2,
    drift_band="0.05_0.5",
    amp_mode="ratio",
    amp_ratio=0.03,
    amp_uv=50.0,
    channel_mode="all",
    n_channels=1,
    intermittent=False,
    intermittent_duty=0.3,
    filter_order=2,
    piecewise_segments=8,
    rng=None,
):
    """
    B2 — baseline drift: slow wandering from poor contact / impedance / movement.

    Modes: sin (fixed Hz), sin_band, random_walk, piecewise.
    Channel scope: all, one, subset; optional intermittent (time-limited) application.
    """
    rng = _get_rng(rng)
    x = np.asarray(x, dtype=np.float64)
    n_ch, n_t = x.shape
    mode = mode.lower()

    if mode == "sin":
        raw = _drift_sin(x, fs, float(drift_hz), rng)
    elif mode == "sin_band":
        raw = _drift_sin_band(x, fs, drift_band, rng)
    elif mode == "random_walk":
        raw = _drift_random_walk(x, fs, drift_band, rng, filter_order=filter_order)
    elif mode == "piecewise":
        raw = _drift_piecewise(x, fs, rng, n_segments=piecewise_segments)
    else:
        raise ValueError(f"Unknown drift mode {mode!r}; expected sin, sin_band, random_walk, piecewise")

    scaled = _scale_artifact(x, raw, amp_mode, amp_ratio, amp_uv)
    ch_mask = _channel_mask(n_ch, channel_mode, n_channels, rng)
    time_mask = _intermittent_time_mask(n_t, intermittent_duty, rng) if intermittent else None
    delta = _apply_spatial_temporal_mask(scaled, ch_mask, time_mask=time_mask)
    return (x + delta).astype(x.dtype, copy=False)


def add_baseline_drift(x, fs, amp_ratio=0.03, freq=0.2):
    """Legacy wrapper: single-frequency sinusoidal drift on all channels."""
    return apply_drift_shift(
        x,
        fs,
        mode="sin",
        drift_hz=freq,
        amp_mode="ratio",
        amp_ratio=amp_ratio,
        channel_mode="all",
    )
