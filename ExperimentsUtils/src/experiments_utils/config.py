from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATEGORIES = ("baseline", "environment", "noise", "adversarial")


def _slug_token(s: str) -> str:
    return str(s).replace(".", "p").replace("-", "_")


def _lf_amp_name(p: dict) -> str:
    if p.get("amp_mode", "ratio") == "uv":
        return f"uv{int(p.get('amp_uv', 50))}"
    return f"ratio{p['amp_ratio']:.2f}"


def _lf_channel_name(p: dict) -> str:
    mode = p.get("channel_mode", "all")
    if mode == "subset":
        return f"subset{int(p.get('n_channels', 1))}"
    return str(mode)


@dataclass
class ExperimentConfig:
    category: str = "baseline"
    shift: str = "none"
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    mcd: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, got {self.category!r}")
        if not self.name:
            self.name = self.default_name()

    def _mcd_name_suffix(self) -> str:
        """Distinct folder suffix when MCD is enabled (appended to shift base name)."""
        if not self.mcd:
            return ""
        parts = ["MCD"]
        parts.append(f"s{int(self.mcd.get('samples', 20))}")
        thresh = float(self.mcd.get("thresh", 0.8))
        if thresh != 0.8:
            parts.append(f"t{str(thresh).replace('.', 'p')}")
        subset = self.mcd.get("subset")
        if subset is not None:
            mode = str(self.mcd.get("subset_mode", "random"))
            if mode == "research":
                parts.append(f"research{int(subset)}")
            else:
                parts.append(f"random{int(subset)}")
                seed = int(self.mcd.get("subset_seed", 42))
                if seed != 42:
                    parts.append(f"seed{seed}")
        return "_" + "_".join(parts)

    def default_name(self) -> str:
        return self._shift_base_name() + self._mcd_name_suffix()

    def _shift_base_name(self) -> str:
        p = self.params
        if self.category == "baseline":
            return "Baseline"
        if self.category == "environment":
            if self.shift == "car":
                return "CAR"
            if self.shift == "fs_mismatch":
                return f"FS_MissMatch_{int(p['target_fs'])}"
            if self.shift == "drop_named":
                ch = p.get("drop_names", "channels").replace(",", "").replace(" ", "")
                return f"DROP_{ch}"
            if self.shift == "quant_adc":
                bits = int(p["bits"])
                uv = p.get("adc_fs_uv")
                return f"QUANT_ADC_b{bits}_uv{int(uv)}" if uv else f"QUANT_ADC_b{bits}"
            if self.shift == "quant":
                return f"QUANT_SYM_b{int(p['bits'])}"
            if self.shift == "drift":
                p = self.params
                mode = p.get("drift_mode", "sin")
                inter = "_inter" if p.get("intermittent") else ""
                if mode == "sin":
                    hz = p.get("drift_hz", 0.2)
                    return (
                        f"DRIFT_sin_{_slug_token(hz)}hz_{_lf_amp_name(p)}"
                        f"_{_lf_channel_name(p)}{inter}"
                    )
                band = _slug_token(p.get("drift_band", "0.05_0.5"))
                return (
                    f"DRIFT_{mode}_{band}_{_lf_amp_name(p)}"
                    f"_{_lf_channel_name(p)}{inter}"
                )
            if self.shift == "impedance":
                p = self.params
                band = _slug_token(p.get("impedance_band", "0_1"))
                return f"IMP_{band}_{_lf_amp_name(p)}_{_lf_channel_name(p)}"
            if self.shift == "drop":
                return f"ST_drop_n{int(p['n_drop'])}"
            if self.shift == "bandpass":
                prof = p.get("bandpass_profile", "0.5_40")
                low, high = prof.split("_", 1)
                low_s = low.replace(".", "p")
                high_s = high.replace(".", "p")
                return f"BP_{low_s}_{high_s}"
        if self.category == "noise":
            if self.shift == "awgn":
                return f"AWGN_{int(p['snr_db'])}"
            if self.shift == "mains":
                hz = int(p.get("mains_hz", 60))
                return f"MAINS{hz}hz_amp{p['amp_ratio']:.2f}"
            if self.shift == "impulse":
                return (
                    f"ST_impulse_den{p['imp_density']}_amp{p['imp_amp']}"
                    f"_burst{int(p.get('imp_burst', 1))}_sym{int(p.get('imp_sym', 1))}"
                )
            if self.shift == "notch":
                profile = p.get("notch_profile", "off")
                if p.get("notch_mode", "only") == "mismatch":
                    mhz = int(p.get("mains_hz", 60))
                    return (
                        f"NOTCH_mismatch_mains{mhz}_notch{profile}"
                        f"_amp{p['amp_ratio']:.2f}"
                    )
                return f"NOTCH_{profile}"
        if self.category == "adversarial":
            mode = p.get("attack_mode", "increase")
            short = "inc" if mode.startswith("inc") else "dec" if mode.startswith("dec") else mode[:4]
            trs = p.get("thresh", 0.5)
            eps = p["eps"]
            if self.shift == "fgsm":
                return f"ST_fgsm_{short}_eps{eps:.2f}_trs{trs:.1f}".replace(".0_", "_")
            if self.shift == "pgd":
                return (
                    f"ST_pgd_{short}_eps{eps:.2f}_alp{p['alpha']:.2f}"
                    f"_stp{int(p['steps'])}_trs{trs:.1f}"
                ).replace(".0_", "_")
        return f"{self.category}_{self.shift}"

    def output_subdir(self) -> str:
        self.validate()
        if self.category == "baseline":
            return str(Path("Results") / "Baseline" / self.name)
        return str(Path("Results") / self.category.capitalize() / self.name)

    def _mcd_env(self) -> dict[str, str]:
        if not self.mcd:
            return {}
        env = {
            "ENABLE_MCD": "1",
            "MCD_SAMPLES": str(self.mcd.get("samples", 20)),
            "MCD_THRESH": str(self.mcd.get("thresh", 0.8)),
            "MCD_SUBSET_SEED": str(self.mcd.get("subset_seed", 42)),
        }
        subset = self.mcd.get("subset")
        if subset is not None:
            env["MCD_SUBSET"] = str(subset)
        subset_mode = self.mcd.get("subset_mode", "random")
        if subset is not None:
            env["MCD_SUBSET_MODE"] = str(subset_mode)
        return env

    def to_env(self) -> dict[str, str]:
        self.validate()
        env: dict[str, str] = {}
        p = self.params
        if self.category == "environment":
            env["SHIFT"] = self.shift
            if self.shift == "fs_mismatch":
                env["TARGET_FS"] = str(p["target_fs"])
            elif self.shift == "drop_named":
                env["DROP_NAMES"] = p["drop_names"]
            elif self.shift == "quant_adc":
                env["BITS"] = str(p["bits"])
                if p.get("adc_fs_uv") is not None:
                    env["ADC_FS_UV"] = str(p["adc_fs_uv"])
            elif self.shift == "quant":
                env["BITS"] = str(p["bits"])
            elif self.shift == "drift":
                p = self.params
                env["DRIFT_MODE"] = p.get("drift_mode", "sin")
                env["AMP_MODE"] = p.get("amp_mode", "ratio")
                if p.get("amp_mode", "ratio") == "uv":
                    env["AMP_UV"] = str(p.get("amp_uv", 50))
                else:
                    env["AMP_RATIO"] = str(p.get("amp_ratio", 0.03))
                env["DRIFT_HZ"] = str(p.get("drift_hz", 0.2))
                env["DRIFT_BAND"] = p.get("drift_band", "0.05_0.5")
                env["CHANNEL_MODE"] = p.get("channel_mode", "all")
                env["N_CHANNELS"] = str(p.get("n_channels", 1))
                env["INTERMITTENT"] = "1" if p.get("intermittent") else "0"
                env["INTERMITTENT_DUTY"] = str(p.get("intermittent_duty", 0.3))
                env["LOWFREQ_ORDER"] = str(p.get("lowfreq_order", 2))
                env["PIECEWISE_SEGMENTS"] = str(p.get("piecewise_segments", 8))
                if p.get("rng_seed") is not None:
                    env["RNG_SEED"] = str(p["rng_seed"])
            elif self.shift == "impedance":
                p = self.params
                env["IMPEDANCE_BAND"] = p.get("impedance_band", "0_1")
                env["AMP_MODE"] = p.get("amp_mode", "ratio")
                if p.get("amp_mode", "ratio") == "uv":
                    env["AMP_UV"] = str(p.get("amp_uv", 50))
                else:
                    env["AMP_RATIO"] = str(p.get("amp_ratio", 0.25))
                env["CHANNEL_MODE"] = p.get("channel_mode", "all")
                env["N_CHANNELS"] = str(p.get("n_channels", 1))
                env["LOWFREQ_ORDER"] = str(p.get("lowfreq_order", 2))
                if p.get("rng_seed") is not None:
                    env["RNG_SEED"] = str(p["rng_seed"])
            elif self.shift == "drop":
                env["N_DROP"] = str(p["n_drop"])
            elif self.shift == "bandpass":
                env["BANDPASS_PROFILE"] = p.get("bandpass_profile", "0.5_40")
                env["BANDPASS_ORDER"] = str(p.get("bandpass_order", 3))
        elif self.category == "noise":
            env["PERTURB"] = self.shift
            if self.shift == "awgn":
                env["SNR_DB"] = str(p["snr_db"])
            elif self.shift == "mains":
                env["AMP_RATIO"] = str(p["amp_ratio"])
                env["MAINS_HZ"] = str(p.get("mains_hz", 60))
            elif self.shift == "impulse":
                env["IMP_DENSITY"] = str(p["imp_density"])
                env["IMP_AMP"] = str(p["imp_amp"])
                env["IMP_BURST"] = str(p.get("imp_burst", 1))
                env["IMP_SYM"] = str(int(p.get("imp_sym", 1)))
                if p.get("rng_seed") is not None:
                    env["RNG_SEED"] = str(p["rng_seed"])
                if p.get("debug_pert"):
                    env["DEBUG_PERT"] = "1"
            elif self.shift == "notch":
                env["NOTCH_PROFILE"] = p.get("notch_profile", "off")
                env["NOTCH_MODE"] = p.get("notch_mode", "only")
                env["NOTCH_Q"] = str(p.get("notch_q", 30.0))
                if p.get("notch_mode", "only") == "mismatch":
                    env["AMP_RATIO"] = str(p["amp_ratio"])
                    env["MAINS_HZ"] = str(p.get("mains_hz", 60))
        elif self.category == "adversarial":
            env["ATTACK"] = self.shift
            env["ATTACK_MODE"] = p.get("attack_mode", "increase")
            env["EPS"] = str(p["eps"])
            env["THRESH"] = str(p.get("thresh", 0.5))
            env["OPEN_K"] = str(p.get("open_k", 5))
            env["CLOSE_K"] = str(p.get("close_k", 5))
            env["MIN_EVENT_SEC"] = str(p.get("min_event_sec", 2.0))
            if self.shift == "pgd":
                env["ALPHA"] = str(p.get("alpha", 0.01))
                env["STEPS"] = str(p.get("steps", 10))
        env.update(self._mcd_env())
        return env

    def to_flat_record(self, data_dir: str, output_dir: str, metrics: dict[str, float]) -> dict[str, Any]:
        self.validate()
        row: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_name": self.name,
            "category": self.category,
            "shift": self.shift,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "params_json": json.dumps(self.params, sort_keys=True),
        }
        row.update(metrics)
        return row

    @classmethod
    def baseline(cls, name: str = "Baseline") -> ExperimentConfig:
        return cls(category="baseline", shift="none", name=name, params={})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
