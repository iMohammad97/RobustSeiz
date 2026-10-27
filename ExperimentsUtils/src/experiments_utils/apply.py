import os

from experiments_utils import shifts


def apply_shifts_from_env(data, fs, labels=None, raw_data=None):
    """Apply environment and noise shifts based on process environment variables."""
    shift = os.getenv("SHIFT", "").lower()
    if shift == "car":
        data = shifts.reref_car(data)
    elif shift == "fs_mismatch":
        try:
            target_fs = float(os.getenv("TARGET_FS", "0"))
        except ValueError:
            target_fs = 0.0
        if target_fs > 0:
            data = shifts.fs_mismatch_roundtrip(data, fs=fs, target_fs=target_fs)
    elif shift == "drop_named":
        names = [s.strip() for s in os.getenv("DROP_NAMES", "").split(",") if s.strip()]
        data = shifts.drop_by_names(data, labels, names)
    elif shift == "quant_adc":
        bits = int(os.getenv("BITS", "12"))
        fs_uv_env = os.getenv("ADC_FS_UV", "")
        fullscale_uv = float(fs_uv_env) if fs_uv_env not in ("", None) else None
        data = shifts.quantize_adc(data, bits=bits, fullscale_uv=fullscale_uv)
    elif shift == "quant":
        b = int(os.getenv("BITS", "12"))
        data = shifts.quantize(data, bits=b)
    elif shift == "drift":
        amp_mode = os.getenv("AMP_MODE", "ratio").lower()
        seed = os.getenv("RNG_SEED", "")
        seed = int(seed) if seed not in ("", None) else None
        data = shifts.apply_drift_shift(
            data,
            fs=fs,
            mode=os.getenv("DRIFT_MODE", "sin").lower(),
            drift_hz=float(os.getenv("DRIFT_HZ", "0.2")),
            drift_band=os.getenv("DRIFT_BAND", "0.05_0.5"),
            amp_mode=amp_mode,
            amp_ratio=float(os.getenv("AMP_RATIO", "0.03")),
            amp_uv=float(os.getenv("AMP_UV", "50")),
            channel_mode=os.getenv("CHANNEL_MODE", "all").lower(),
            n_channels=int(os.getenv("N_CHANNELS", "1")),
            intermittent=os.getenv("INTERMITTENT", "0") == "1",
            intermittent_duty=float(os.getenv("INTERMITTENT_DUTY", "0.3")),
            filter_order=int(os.getenv("LOWFREQ_ORDER", "2")),
            piecewise_segments=int(os.getenv("PIECEWISE_SEGMENTS", "8")),
            rng=seed,
        )
        print(
            f"[SHIFT] drift mode={os.getenv('DRIFT_MODE', 'sin')} "
            f"band={os.getenv('DRIFT_BAND', '')} amp_mode={amp_mode} "
            f"channels={os.getenv('CHANNEL_MODE', 'all')}"
        )
    elif shift == "impedance":
        amp_mode = os.getenv("AMP_MODE", "ratio").lower()
        seed = os.getenv("RNG_SEED", "")
        seed = int(seed) if seed not in ("", None) else None
        band = os.getenv("IMPEDANCE_BAND", "0_1")
        data = shifts.apply_impedance_shift(
            data,
            fs=fs,
            band=band,
            amp_mode=amp_mode,
            amp_ratio=float(os.getenv("AMP_RATIO", "0.25")),
            amp_uv=float(os.getenv("AMP_UV", "50")),
            channel_mode=os.getenv("CHANNEL_MODE", "all").lower(),
            n_channels=int(os.getenv("N_CHANNELS", "1")),
            filter_order=int(os.getenv("LOWFREQ_ORDER", "2")),
            rng=seed,
        )
        print(
            f"[SHIFT] impedance band={band} amp_mode={amp_mode} "
            f"channels={os.getenv('CHANNEL_MODE', 'all')}"
        )
    elif shift == "drop":
        n = int(os.getenv("N_DROP", "1"))
        data = shifts.drop_channels(data, n_drop=n)
    elif shift == "bandpass":
        profile = os.getenv("BANDPASS_PROFILE", "0.5_40")
        order = int(os.getenv("BANDPASS_ORDER", "3"))
        data = shifts.apply_bandpass_shift(data, fs=fs, profile=profile, order=order)
        low, high = shifts.parse_bandpass_profile(profile)
        print(f"[SHIFT] bandpass {low}-{high} Hz order={order}")

    mode = os.environ.get("PERTURB", "").lower()
    seed = os.getenv("RNG_SEED", "")
    seed = int(seed) if seed not in ("", None) else None

    if mode == "awgn":
        snr = float(os.environ.get("SNR_DB", "20"))
        data = shifts.add_awgn_snr(data, snr_db=snr, rng=seed)
        print(f"Added AWGN with SNR={snr}dB")
    elif mode == "mains":
        amp = float(os.environ.get("AMP_RATIO", "0.03"))
        data = shifts.add_mains(
            data, fs=fs, amp_ratio=amp, freq=float(os.environ.get("MAINS_HZ", "60")), rng=seed
        )
    elif mode == "impulse":
        density = float(os.getenv("IMP_DENSITY", "0.001"))
        amp = float(os.getenv("IMP_AMP", "3.0"))
        burst = int(os.getenv("IMP_BURST", "1"))
        symmetric = os.getenv("IMP_SYM", "1") != "0"
        data = shifts.add_impulse_noise(
            data,
            density=density,
            amp_scale=amp,
            burst_len=burst,
            symmetric=symmetric,
            rng=seed,
        )
        if os.getenv("DEBUG_PERT", "0") == "1" and raw_data is not None:
            rms = lambda a: float(shifts._rms(a))
            print(
                f"[PERTURB] {mode} dens={density} amp={amp} burst={burst} sym={symmetric} "
                f"rms_before={rms(raw_data):.3f} rms_after={rms(data):.3f}"
            )
    elif mode == "notch":
        profile = os.environ.get("NOTCH_PROFILE", "off")
        notch_mode = os.environ.get("NOTCH_MODE", "only").lower()
        amp = float(os.environ.get("AMP_RATIO", "0.03"))
        mains_hz = float(os.environ.get("MAINS_HZ", "60"))
        q = float(os.environ.get("NOTCH_Q", "30"))
        data = shifts.apply_notch_shift(
            data,
            fs=fs,
            profile=profile,
            mode=notch_mode,
            mains_hz=mains_hz,
            amp_ratio=amp,
            q=q,
            rng=seed,
        )
        print(
            f"[PERTURB] notch profile={profile} mode={notch_mode} "
            f"mains_hz={mains_hz} amp_ratio={amp} q={q}"
        )

    return data
