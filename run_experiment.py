#!/usr/bin/env python3
"""End-to-end robustness experiment runner: shift → infer → evaluate → registry."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for _src in ("ExperimentsUtils/src", "Evaluation/src"):
    _p = PROJECT_ROOT / _src
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments_utils.config import ExperimentConfig  # noqa: E402
from experiments_utils.runner import run_docker_inference  # noqa: E402


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    from experiments_utils.datasets import DATASETS

    parser.add_argument(
        "--dataset",
        default="TUSZ",
        choices=list(DATASETS),
        help="Evaluation dataset (sets the default --data-dir and is logged in the registry).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model to run (default: ACTIVE_MODEL env var, else Model/active_model).",
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Eval dataset root (BIDS EDFs + reference annotations; default: the dataset's folder).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Where to store predictions and per-experiment results (default: auto under Results/).",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Experiment folder name (default: auto from parameters).",
    )
    parser.add_argument(
        "--results-registry",
        default="experiments_results.csv",
        help="CSV file at repo root logging all experiment metrics and parameters.",
    )
    parser.add_argument(
        "--runner",
        default="docker",
        choices=["docker", "local"],
        help="docker: one container per EDF (default); local: in-process in the current Python env.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Local runner: skip recordings whose outputs already exist.",
    )
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")

    parser.add_argument(
        "--enable-mcd",
        action="store_true",
        help="Enable Monte Carlo dropout uncertainty (optional; requires model MCD support).",
    )
    parser.add_argument(
        "--mcd-samples",
        type=int,
        default=20,
        help="Number of stochastic dropout forwards per recording (default: 20).",
    )
    parser.add_argument(
        "--mcd-thresh",
        type=float,
        default=0.8,
        help="Decision threshold for agreement index (default: 0.8).",
    )
    parser.add_argument(
        "--mcd-subset",
        type=int,
        default=None,
        help="When MCD is enabled, limit inference scope (see --mcd-subset-mode).",
    )
    parser.add_argument(
        "--mcd-subset-mode",
        default="random",
        choices=["random", "research"],
        help=(
            "How --mcd-subset is interpreted: random=N EDFs (seeded); "
            "research=first N sub-* dirs and all their sessions (paper demo)."
        ),
    )
    parser.add_argument(
        "--mcd-subset-seed",
        type=int,
        default=42,
        help="RNG seed for --mcd-subset-mode random (default: 42).",
    )
    parser.add_argument(
        "--research-subset",
        type=int,
        default=None,
        help="Without MCD: run and score only the first N subjects (research-subset mode).",
    )
    parser.add_argument(
        "--onset-subset",
        type=int,
        default=None,
        help=(
            "Limit lead/lag onset scoring scope (see --onset-subset-mode). "
            "Default: when MCD uses --mcd-subset, onset inherits that scope; "
            "otherwise full eval. Explicit value always wins for onset."
        ),
    )
    parser.add_argument(
        "--onset-subset-mode",
        default="random",
        choices=["random", "research"],
        help=(
            "How --onset-subset is interpreted: random=N reference files (seeded); "
            "research=first N sub-* dirs and all their sessions."
        ),
    )
    parser.add_argument(
        "--onset-subset-seed",
        type=int,
        default=42,
        help="RNG seed for --onset-subset-mode random (default: 42).",
    )


def _onset_subset_config(args: argparse.Namespace):
    from evaluation.subset import EvalSubsetConfig

    if args.onset_subset is None:
        return None
    return EvalSubsetConfig(
        subset=args.onset_subset,
        subset_mode=args.onset_subset_mode,
        subset_seed=args.onset_subset_seed,
    )


def _eval_subset_from_mcd(cfg: ExperimentConfig):
    """Match SzCORE (+ default onset) to MCD inference subset when present."""
    from evaluation.subset import EvalSubsetConfig

    if cfg.subset and not cfg.mcd:
        return EvalSubsetConfig(subset=int(cfg.subset), subset_mode="research")
    if not cfg.mcd or cfg.mcd.get("subset") is None:
        return EvalSubsetConfig()
    return EvalSubsetConfig(
        subset=int(cfg.mcd["subset"]),
        subset_mode=str(cfg.mcd.get("subset_mode", "random")),
        subset_seed=int(cfg.mcd.get("subset_seed", 42)),
    )


def _apply_onset_subset_params(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.onset_subset is None:
        return cfg
    cfg.params = {
        **cfg.params,
        "onset_subset": args.onset_subset,
        "onset_subset_mode": args.onset_subset_mode,
        "onset_subset_seed": args.onset_subset_seed,
    }
    return cfg


def _apply_mcd_config(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if not args.enable_mcd:
        return cfg
    mcd = {
        "samples": args.mcd_samples,
        "thresh": args.mcd_thresh,
        "subset_seed": args.mcd_subset_seed,
        "subset_mode": args.mcd_subset_mode,
    }
    if args.mcd_subset is not None:
        mcd["subset"] = args.mcd_subset
    cfg.mcd = mcd
    cfg.params = {
        **cfg.params,
        "mcd_enabled": True,
        "mcd_samples": args.mcd_samples,
        "mcd_thresh": args.mcd_thresh,
        "mcd_subset_seed": args.mcd_subset_seed,
        "mcd_subset_mode": args.mcd_subset_mode,
    }
    if args.mcd_subset is not None:
        cfg.params["mcd_subset"] = args.mcd_subset
        # Demo default: SzCORE + onset scored on the same inference subset.
        cfg.params["eval_subset"] = args.mcd_subset
        cfg.params["eval_subset_mode"] = args.mcd_subset_mode
        cfg.params["eval_subset_seed"] = args.mcd_subset_seed
    return cfg


def _lowfreq_params(args: argparse.Namespace) -> dict:
    params: dict = {
        "amp_mode": args.amp_mode,
        "channel_mode": args.channel_mode,
        "n_channels": args.n_channels,
        "lowfreq_order": args.lowfreq_order,
    }
    if args.amp_mode == "uv":
        params["amp_uv"] = args.amp_uv
    else:
        params["amp_ratio"] = args.amp_ratio
    if args.rng_seed is not None:
        params["rng_seed"] = args.rng_seed
    return params


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    params: dict = {}
    category = args.category
    shift = args.shift

    if category == "baseline":
        return ExperimentConfig.baseline(name=args.name or "Baseline")

    if category == "environment":
        if shift == "none":
            raise ValueError(
                "environment experiments require --shift "
                "(car, fs_mismatch, drop_named, drop, quant_adc, quant, drift, impedance, bandpass)"
            )
        if shift == "fs_mismatch":
            if args.target_fs is None:
                raise ValueError("--target-fs is required for fs_mismatch")
            params["target_fs"] = args.target_fs
        elif shift == "drop_named":
            params["drop_names"] = args.drop_names
        elif shift == "quant_adc":
            params["bits"] = args.bits
            if args.adc_fs_uv is not None:
                params["adc_fs_uv"] = args.adc_fs_uv
        elif shift == "quant":
            params["bits"] = args.bits
        elif shift == "drift":
            params = _lowfreq_params(args)
            params.update(
                {
                    "drift_mode": args.drift_mode,
                    "drift_hz": args.drift_hz,
                    "drift_band": args.drift_band,
                    "intermittent": args.intermittent,
                    "intermittent_duty": args.intermittent_duty,
                    "piecewise_segments": args.piecewise_segments,
                }
            )
        elif shift == "impedance":
            params = _lowfreq_params(args)
            params["impedance_band"] = args.impedance_band
            if args.amp_mode == "ratio" and args.amp_ratio == 0.03:
                params["amp_ratio"] = 0.25
        elif shift == "drop":
            params["n_drop"] = args.n_drop
        elif shift == "bandpass":
            params["bandpass_profile"] = args.bandpass_profile
            params["bandpass_order"] = args.bandpass_order
        cfg = ExperimentConfig(category=category, shift=shift, params=params)
    elif category == "noise":
        if shift == "none":
            raise ValueError("noise experiments require --shift (awgn, mains, impulse, notch)")
        if shift in ("awgn", "mains", "notch") and args.rng_seed is not None:
            params["rng_seed"] = args.rng_seed
        if shift == "awgn":
            params["snr_db"] = args.snr_db
        elif shift == "mains":
            params["amp_ratio"] = args.amp_ratio
            params["mains_hz"] = args.mains_hz
        elif shift == "impulse":
            params.update(
                {
                    "imp_density": args.imp_density,
                    "imp_amp": args.imp_amp,
                    "imp_burst": args.imp_burst,
                    "imp_sym": args.imp_sym,
                    "rng_seed": args.rng_seed,
                    "debug_pert": args.debug_pert,
                }
            )
        elif shift == "notch":
            params.update(
                {
                    "notch_profile": args.notch_profile,
                    "notch_mode": args.notch_mode,
                    "notch_q": args.notch_q,
                    "amp_ratio": args.amp_ratio,
                    "mains_hz": args.mains_hz,
                }
            )
        cfg = ExperimentConfig(category=category, shift=shift, params=params)
    elif category == "adversarial":
        if shift not in ("fgsm", "pgd"):
            raise ValueError("adversarial experiments require --shift fgsm or pgd")
        params.update(
            {
                "attack_mode": args.attack_mode,
                "eps": args.eps,
                "thresh": args.thresh,
                "open_k": args.open_k,
                "close_k": args.close_k,
                "min_event_sec": args.min_event_sec,
            }
        )
        if shift == "pgd":
            params["alpha"] = args.alpha
            params["steps"] = args.steps
        cfg = ExperimentConfig(category=category, shift=shift, params=params)
    else:
        raise ValueError(f"Unknown category {category!r}")

    if args.name:
        cfg.name = args.name
    cfg.validate()
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a seizure robustness experiment (Docker inference + SzCORE evaluation)."
    )
    _add_shared_args(parser)
    parser.add_argument(
        "--category",
        required=True,
        choices=["baseline", "environment", "noise", "adversarial"],
    )
    parser.add_argument("--shift", default="none", help="Shift type within the category.")

    parser.add_argument("--target-fs", type=float, help="Environment: fs_mismatch target Hz.")
    parser.add_argument("--drop-names", default="T7,T8", help="Environment: comma-separated channels.")
    parser.add_argument("--bits", type=int, default=12, help="Quantization bit depth.")
    parser.add_argument("--adc-fs-uv", type=float, default=None, help="ADC full-scale µV.")

    parser.add_argument("--snr-db", type=float, default=20, help="AWGN SNR in dB.")
    parser.add_argument(
        "--amp-mode",
        default="ratio",
        choices=["ratio", "uv"],
        help="Environment drift/impedance: amplitude as fraction of channel std (ratio) or fixed µV (uv).",
    )
    parser.add_argument(
        "--amp-uv",
        type=float,
        default=50.0,
        choices=[25.0, 50.0, 100.0, 200.0],
        help="Environment drift/impedance: fixed peak amplitude in µV when --amp-mode uv.",
    )
    parser.add_argument(
        "--amp-ratio",
        type=float,
        default=0.03,
        help="Amplitude ratio vs channel std (drift/impedance/mains); impedance uses 0.25 if left at default 0.03.",
    )
    parser.add_argument("--mains-hz", type=float, default=60)
    parser.add_argument("--drift-hz", type=float, default=0.2, help="Environment/drift sin mode: frequency in Hz.")
    parser.add_argument(
        "--drift-mode",
        default="sin",
        choices=["sin", "sin_band", "random_walk", "piecewise"],
        help="Environment/drift: waveform type.",
    )
    parser.add_argument(
        "--drift-band",
        default="0.05_0.5",
        choices=["0_1", "0.05_0.5", "0.1_1"],
        help="Environment/drift: band for sin_band / random_walk.",
    )
    parser.add_argument(
        "--impedance-band",
        default="0_1",
        choices=["0_1", "0.05_0.5", "0.1_1"],
        help="Environment/impedance: slow noise band (0_1 = LP 1 Hz).",
    )
    parser.add_argument(
        "--channel-mode",
        default="all",
        choices=["all", "one", "subset"],
        help="Environment drift/impedance: which channels receive the artifact.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=2,
        help="Environment drift/impedance: channels when --channel-mode subset.",
    )
    parser.add_argument(
        "--intermittent",
        action="store_true",
        help="Environment/drift: apply drift only during random time segments.",
    )
    parser.add_argument(
        "--intermittent-duty",
        type=float,
        default=0.3,
        help="Environment/drift intermittent: fraction of recording with drift active.",
    )
    parser.add_argument(
        "--lowfreq-order",
        type=int,
        default=2,
        help="Environment drift/impedance: Butterworth order for slow-noise filtering.",
    )
    parser.add_argument(
        "--piecewise-segments",
        type=int,
        default=8,
        help="Environment/drift piecewise mode: number of drift segments.",
    )
    parser.add_argument("--imp-density", type=float, default=0.001)
    parser.add_argument("--imp-amp", type=float, default=3.0)
    parser.add_argument("--imp-burst", type=int, default=1)
    parser.add_argument("--imp-sym", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--debug-pert", action="store_true")
    parser.add_argument("--n-drop", type=int, default=1, help="Environment: random channels to zero.")
    parser.add_argument(
        "--bandpass-profile",
        default="0.5_40",
        choices=["0.5_40", "1_40", "1_30", "0.5_25"],
        help="Environment/bandpass: lower_upper Hz (e.g. 0.5_40 = 0.5–40 Hz on raw EEG).",
    )
    parser.add_argument(
        "--bandpass-order",
        type=int,
        default=3,
        help="Environment/bandpass: Butterworth filter order (default 3).",
    )

    parser.add_argument(
        "--notch-profile",
        default="off",
        choices=["off", "50", "60", "50_100", "60_120"],
        help="Noise/notch: off, 50, 60, 50_100 (50+100 Hz), 60_120 (60+120 Hz).",
    )
    parser.add_argument(
        "--notch-mode",
        default="only",
        choices=["only", "mismatch"],
        help="Noise/notch: only=notch on raw EEG; mismatch=inject mains then notch (powerline mismatch).",
    )
    parser.add_argument("--notch-q", type=float, default=30.0, help="Noise/notch: IIR notch Q factor.")

    parser.add_argument(
        "--attack-mode",
        default="increase",
        choices=["increase", "decrease", "flip"],
    )
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--open-k", type=int, default=5)
    parser.add_argument("--close-k", type=int, default=5)
    parser.add_argument("--min-event-sec", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _apply_onset_subset_params(_apply_mcd_config(_build_config(args), args), args)
    if args.research_subset is not None:
        if args.enable_mcd:
            raise SystemExit("Use --mcd-subset with --enable-mcd; --research-subset is for runs without MCD.")
        cfg.subset = args.research_subset
        cfg.params = {**cfg.params, "eval_subset": args.research_subset, "eval_subset_mode": "research"}
    if not args.name:
        cfg.name = cfg.default_name()

    from experiments_utils.datasets import get_dataset
    from experiments_utils.models.registry import get_active_model_name

    if args.model:
        os.environ["ACTIVE_MODEL"] = args.model
    cfg.model = get_active_model_name(PROJECT_ROOT)
    dataset = get_dataset(args.dataset)
    cfg.dataset = dataset.name

    data_dir = (PROJECT_ROOT / (args.data_dir or dataset.data_dir)).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (PROJECT_ROOT / cfg.output_subdir()).resolve()

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    print(f"Experiment: {cfg.name}")
    print(f"  model={cfg.model} dataset={cfg.dataset}")
    print(f"  category={cfg.category} shift={cfg.shift}")
    if cfg.mcd:
        subset_info = ""
        if cfg.mcd.get("subset") is not None:
            mode = cfg.mcd.get("subset_mode", "random")
            if mode == "research":
                subset_info = f" research-subset={cfg.mcd.get('subset')} subjects"
            else:
                subset_info = (
                    f" random-subset={cfg.mcd.get('subset')} "
                    f"seed={cfg.mcd.get('subset_seed')}"
                )
        print(
            f"  mcd=samples:{cfg.mcd.get('samples')} thresh:{cfg.mcd.get('thresh')}{subset_info}"
        )
    if args.onset_subset is not None:
        print(
            f"  onset-subset={args.onset_subset} mode={args.onset_subset_mode} "
            f"seed={args.onset_subset_seed}"
        )
    eval_subset = _eval_subset_from_mcd(cfg)
    if eval_subset.is_active():
        print(
            f"  eval-subset={eval_subset.subset} mode={eval_subset.subset_mode} "
            f"(SzCORE + default onset matched to MCD inference scope)"
        )
    print(f"  data={data_dir}")
    print(f"  output={output_dir}")

    if cfg.mcd:
        from experiments_utils.models.registry import get_active_model_name, model_supports_mcd

        if not model_supports_mcd(PROJECT_ROOT):
            model_name = get_active_model_name(PROJECT_ROOT)
            print(
                f"WARNING: --enable-mcd set but model {model_name!r} does not support MCD; "
                "performance metrics will still run, uncertainty will be skipped.",
                file=sys.stderr,
            )

    if not args.skip_inference:
        env = {**cfg.to_env(), "ACTIVE_MODEL": cfg.model}
        if args.runner == "local":
            from experiments_utils.runner import run_local_inference

            run_local_inference(data_dir, output_dir, env, PROJECT_ROOT, resume=args.resume)
        else:
            run_docker_inference(
                data_dir=data_dir,
                output_dir=output_dir,
                env=env,
                project_root=PROJECT_ROOT,
            )

    from experiments_utils.runner import write_run_status

    metrics: dict[str, float] = {}
    if not args.skip_evaluate:
        from evaluation.evaluate import evaluate
        from evaluation.registry import append_experiment_result

        write_run_status(state="scoring")
        from experiments_utils.models.registry import model_supports_mcd

        metrics = evaluate(
            str(data_dir),
            str(output_dir),
            output_dir=str(output_dir),
            eval_subset=eval_subset,
            onset_subset=_onset_subset_config(args),
            dataset=cfg.dataset,
        )
        if cfg.mcd and model_supports_mcd(PROJECT_ROOT):
            from evaluation.uncertainty import evaluate_uncertainty

            metrics.update(
                evaluate_uncertainty(
                    str(data_dir),
                    str(output_dir),
                    output_dir=str(output_dir),
                    eval_subset=eval_subset,
                )
            )
        record = cfg.to_flat_record(str(data_dir), str(output_dir), metrics)
        registry_path = (PROJECT_ROOT / args.results_registry).resolve()
        append_experiment_result(registry_path, record)
        write_run_status(
            state="done",
            event_f1=metrics.get("event_f1"),
            event_sensitivity=metrics.get("event_sensitivity"),
            event_fp_per_day=metrics.get("event_fp_per_day"),
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
