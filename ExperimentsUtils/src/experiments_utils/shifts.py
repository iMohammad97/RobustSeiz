import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import butter, iirnotch, lfilter, resample

NOTCH_PROFILES: dict[str, list[float]] = {
    "off": [],
    "50": [50.0],
    "60": [60.0],
    "50_100": [50.0, 100.0],
    "60_120": [60.0, 120.0],
}

# (low_hz, high_hz) acquisition-style band-pass presets on raw µV
BANDPASS_PROFILES: dict[str, tuple[float, float]] = {
    "0.5_40": (0.5, 40.0),
    "1_40": (1.0, 40.0),
    "1_30": (1.0, 30.0),
    "0.5_25": (0.5, 25.0),
}


def _rms(x, axis=None, keepdims=False):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(np.square(x), axis=axis, keepdims=keepdims))


def add_awgn_snr(x, snr_db, rng=None):
    if snr_db is None or np.isinf(snr_db):
        return x
    rng = np.random.default_rng(rng)
    s_rms = _rms(x, axis=1, keepdims=True)
    n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    noise = rng.standard_normal(x.shape)
    noise_rms = _rms(noise, axis=1, keepdims=True) + 1e-12
    noise = noise / noise_rms * n_rms
    return (x + noise).astype(x.dtype, copy=False)


def parse_notch_profile(profile: str) -> list[float]:
    key = profile.strip().lower().replace("-", "_")
    if key not in NOTCH_PROFILES:
        raise ValueError(f"Unknown notch profile {profile!r}; expected one of {tuple(NOTCH_PROFILES)}")
    return list(NOTCH_PROFILES[key])


def apply_notch_filter(x, fs, freqs, q=30.0):
    """Apply IIR notch filters per channel on raw µV data (before model band-pass)."""
    if not freqs:
        return x
    y = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    for freq in freqs:
        if freq <= 0 or freq >= nyq:
            raise ValueError(f"Notch frequency {freq} Hz must be in (0, Nyquist={nyq:.1f})")
        b, a = iirnotch(freq, Q=q, fs=fs)
        y = lfilter(b, a, y, axis=1)
    return y.astype(x.dtype, copy=False)


def apply_notch_shift(x, fs, profile="off", mode="only", mains_hz=60.0, amp_ratio=0.03, q=30.0, rng=None):
    """
    Benchmark notch preprocessing on raw EEG.

    mode=only: apply notch profile only (off = pass-through).
    mode=mismatch: inject mains hum, then apply notch (wrong-line preprocessing test).
    """
    freqs = parse_notch_profile(profile)
    mode = mode.lower()
    if mode == "mismatch":
        x = add_mains(x, fs=fs, amp_ratio=amp_ratio, freq=mains_hz, rng=rng)
        if freqs:
            x = apply_notch_filter(x, fs=fs, freqs=freqs, q=q)
    elif mode == "only":
        if freqs:
            x = apply_notch_filter(x, fs=fs, freqs=freqs, q=q)
    else:
        raise ValueError(f"Unknown notch mode {mode!r}; expected 'only' or 'mismatch'")
    return x


def parse_bandpass_profile(profile: str) -> tuple[float, float]:
    key = profile.strip().lower().replace("-", "_")
    if key not in BANDPASS_PROFILES:
        raise ValueError(
            f"Unknown band-pass profile {profile!r}; expected one of {tuple(BANDPASS_PROFILES)}"
        )
    return BANDPASS_PROFILES[key]


def apply_bandpass_filter(x, fs, lowcut, highcut, order=3):
    """Butterworth band-pass on raw µV (per channel), before model preprocessing."""
    nyq = fs / 2.0
    if lowcut <= 0 or highcut <= lowcut or highcut >= nyq:
        raise ValueError(
            f"Band-pass cutoffs must satisfy 0 < {lowcut} < {highcut} < Nyquist={nyq:.1f}"
        )
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    y = lfilter(b, a, np.asarray(x, dtype=np.float64), axis=1)
    return y.astype(x.dtype, copy=False)


def apply_bandpass_shift(x, fs, profile="0.5_40", order=3):
    lowcut, highcut = parse_bandpass_profile(profile)
    return apply_bandpass_filter(x, fs=fs, lowcut=lowcut, highcut=highcut, order=order)


def add_mains(x, fs, amp_ratio=0.03, freq=60.0, phases=None, rng=None):
    if amp_ratio <= 0:
        return x
    t = np.arange(x.shape[1]) / fs
    if phases is None:
        phases = 2 * np.pi * np.random.default_rng(rng).random(x.shape[0])
    s_rms = _rms(x, axis=1)
    hum = np.array(
        [amp_ratio * s_rms[i] * np.sin(2 * np.pi * freq * t + phases[i]) for i in range(x.shape[0])]
    )
    return x + hum


def add_impulse_noise(x, density=1e-3, amp_scale=3.0, burst_len=1, symmetric=True, rng=None):
    assert x.ndim == 2, "x must be (C, T)"
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    elif rng is None:
        rng = np.random.default_rng()

    amp = amp_scale * _rms(x, axis=1, keepdims=True)
    center_mask = rng.random(x.shape) < float(density)
    if burst_len and burst_len > 1:
        structure = np.ones(burst_len, dtype=bool)
        mask = binary_dilation(center_mask, structure=structure[np.newaxis, :])
    else:
        mask = center_mask

    noise = np.zeros_like(x, dtype=np.float64)
    signs = rng.choice([-1.0, 1.0], size=(x.shape[0], 1)) if symmetric else np.ones((x.shape[0], 1))
    noise[mask] = (amp * signs)[np.repeat(np.arange(x.shape[0]), mask.sum(axis=1)), 0]
    return (x.astype(np.float64, copy=False) + noise).astype(x.dtype, copy=False)


from experiments_utils.lowfreq_shifts import (  # noqa: E402
    IMPEDANCE_AMP_UV_PRESETS,
    LOWFREQ_BANDS,
    add_baseline_drift,
    apply_drift_shift,
    apply_impedance_shift,
    parse_lowfreq_band,
)


def drop_channels(x, n_drop=0, rng=None):
    if n_drop <= 0:
        return x
    rng = np.random.default_rng(rng)
    chans = rng.choice(x.shape[0], size=min(n_drop, x.shape[0]), replace=False)
    x = x.copy()
    x[chans, :] = 0.0
    return x


def quantize(x, bits=12):
    if bits >= 16:
        return x
    xmax = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
    levels = 2 ** (bits - 1) - 1
    return np.round(x / xmax * levels) / levels * xmax


def reref_car(x):
    return x - np.mean(x, axis=0, keepdims=True)


def fs_mismatch_roundtrip(x, fs, target_fs):
    _, t = x.shape
    if target_fs <= 0 or abs(target_fs - fs) < 1e-6:
        return x
    t_mid = max(1, int(round(t * (target_fs / fs))))
    y = resample(x, t_mid, axis=1)
    return resample(y, t, axis=1)


# Old (T3/T4/T5/T6) and new (T7/T8/P7/P8) 10-20 names for the same electrodes.
_ELECTRODE_SYNONYMS = {"t7": "t3", "t8": "t4", "p7": "t5", "p8": "t6", "01": "o1"}
_REFERENCE_TOKENS = {"avg", "ref", "cs2", "le", "ar"}


def _canonical_electrode(name):
    token = str(name).strip().lower().removeprefix("eeg ").strip()
    return _ELECTRODE_SYNONYMS.get(token, token)


def channel_electrodes(label):
    """Electrodes in a channel label: ``FP1-Avg`` -> {fp1}; bipolar ``T7-P7`` -> {t3, t5}."""
    parts = str(label).strip().lower().removeprefix("eeg ").split("-")
    return {_canonical_electrode(p) for p in parts if p.strip() and p.strip() not in _REFERENCE_TOKENS}


def drop_by_names(x, labels, names):
    """Zero every channel that involves one of ``names`` (unipolar or bipolar labels)."""
    if not names:
        return x
    x = x.copy()
    name_set = {_canonical_electrode(n) for n in names if n.strip()}
    if labels is not None:
        for i, label in enumerate(labels):
            if channel_electrodes(label) & name_set:
                x[i, :] = 0.0
    else:
        for n in list(name_set):
            if n.startswith("ch:"):
                try:
                    i = int(n.split(":", 1)[1])
                    if 0 <= i < x.shape[0]:
                        x[i, :] = 0.0
                except ValueError:
                    pass
    return x


def quantize_adc(x, bits=12, fullscale_uv=None):
    if bits >= 16:
        return x
    levels = (2 ** (bits - 1)) - 1
    if fullscale_uv is not None and fullscale_uv > 0:
        xmax = float(fullscale_uv)
        x_clip = np.clip(x, -xmax, xmax)
        return np.round(x_clip / xmax * levels) / levels * xmax
    xmax = np.max(np.abs(x), axis=1, keepdims=True) + 1e-12
    return np.round(x / xmax * levels) / levels * xmax
