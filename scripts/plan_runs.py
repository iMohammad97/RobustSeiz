#!/usr/bin/env python3
"""Benchmark run plan: shift grid x model/dataset coverage, full eval + 10-subject MCD research subset.

    python scripts/plan_runs.py                  # list the runs still needed (FGSM attacks)
    python scripts/plan_runs.py --coverage       # coverage matrix incl. valid existing registry rows
    python scripts/plan_runs.py --attack pgd     # deferred PGD runs (see PROJECT.md)

Run them in parallel with scripts/run_queue.py.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _src in ("ExperimentsUtils/src", "Evaluation/src", "."):
    sys.path.insert(0, str(PROJECT_ROOT / _src))

SEED = ["--rng-seed", "42"]
PGD_STEPS = 10
ATTACK_EPS = (0.01, 0.05, 0.1)


def _attack(kind: str, mode: str, eps: float) -> list[str]:
    args = ["--category", "adversarial", "--shift", kind, "--attack-mode", mode, "--eps", str(eps), "--thresh", "0.5"]
    if kind == "pgd":
        args += ["--alpha", str(round(2.5 * eps / PGD_STEPS, 5)), "--steps", str(PGD_STEPS)]  # Madry et al. step size
    return args


# 31 non-adversarial settings (baseline + 30 shifts); 6 attack settings are added per attack kind.
BASE_GRID: list[list[str]] = [
    ["--category", "baseline"],
    *[["--category", "environment", "--shift", "bandpass", "--bandpass-profile", p] for p in ("0.5_40", "1_40", "0.5_25")],
    *[["--category", "environment", "--shift", "drop_named", "--drop-names", n] for n in ("T7,T8", "O1,O2")],
    *[["--category", "environment", "--shift", "fs_mismatch", "--target-fs", f] for f in ("50", "100", "200")],
    ["--category", "environment", "--shift", "quant_adc", "--bits", "5"],
    ["--category", "environment", "--shift", "quant_adc", "--bits", "10", "--adc-fs-uv", "100"],
    ["--category", "environment", "--shift", "quant_adc", "--bits", "12"],
    *[["--category", "environment", "--shift", "impedance", "--impedance-band", "0_1", "--amp-mode", "uv",
       "--amp-uv", uv, "--channel-mode", "all", *SEED] for uv in ("50", "100", "200")],
    ["--category", "environment", "--shift", "drift", "--drift-mode", "sin", "--drift-hz", "0.2",
     "--amp-mode", "uv", "--amp-uv", "50", "--channel-mode", "all", *SEED],
    ["--category", "environment", "--shift", "drift", "--drift-mode", "random_walk", "--drift-band", "0.05_0.5",
     "--amp-mode", "ratio", "--amp-ratio", "0.10", "--channel-mode", "all", *SEED],
    ["--category", "environment", "--shift", "drift", "--drift-mode", "piecewise", "--drift-band", "0.05_0.5",
     "--amp-mode", "ratio", "--amp-ratio", "0.10", "--channel-mode", "all", "--piecewise-segments", "8", *SEED],
    *[["--category", "noise", "--shift", "awgn", "--snr-db", s, *SEED] for s in ("0", "2", "5", "20", "50")],
    *[["--category", "noise", "--shift", "mains", "--mains-hz", "60", "--amp-ratio", a, *SEED] for a in ("0.1", "0.5", "1.0")],
    *[["--category", "noise", "--shift", "impulse", "--imp-density", d, "--imp-amp", "4.0", "--imp-burst", "1",
       "--imp-sym", "1", *SEED] for d in ("0.0005", "0.005", "0.01")],
    ["--category", "noise", "--shift", "notch", "--notch-mode", "only", "--notch-profile", "50"],
    ["--category", "noise", "--shift", "notch", "--notch-mode", "mismatch", "--mains-hz", "60",
     "--notch-profile", "50", "--amp-ratio", "0.10", *SEED],
]


def grid(attack: str) -> list[list[str]]:
    return BASE_GRID + [_attack(attack, m, e) for m in ("increase", "decrease") for e in ATTACK_EPS]


MCD_SUBSET = ["--enable-mcd", "--mcd-samples", "5", "--mcd-subset", "10", "--mcd-subset-mode", "research", "--mcd-thresh", "0.8"]
PLAIN_SUBSET = ["--research-subset", "10"]

# model -> (held-out, compatible datasets; supports MCD). See Data/README.md and each Model/*/MODEL.md.
COVERAGE: dict[str, tuple[tuple[str, ...], bool]] = {
    "SeizureTransformer": (("TUSZ",), True),
    "EEGNet": (("TUSZ", "Siena", "CHBMIT"), True),
    "EEGWaveNet": (("TUSZ", "Siena"), False),
    "EventNet": (("TUSZ", "Siena"), False),
}

# Rows logged before the 2026-10 fixes are invalid for these families (stale Docker image, no-op
# named drop, CAR no-op, inverted/threshold-leaking attacks). Full-mode band-pass and notch-only
# rows were also identical to baseline; their research-subset (MCD) rows are valid.
LEGACY_CUTOFF = "2026-10-01"
LEGACY_INVALID = {"impedance", "drift", "drop_named", "car", "fgsm", "pgd"}
LEGACY_INVALID_FULL = {"bandpass", "notch_only"}


@dataclass(frozen=True)
class Run:
    model: str
    dataset: str
    subset: bool
    setting: tuple[str, ...]

    @property
    def args(self) -> list[str]:
        extra = (MCD_SUBSET if COVERAGE[self.model][1] else PLAIN_SUBSET) if self.subset else []
        return ["--model", self.model, "--dataset", self.dataset, *self.setting, *extra]

    @property
    def family(self) -> str:
        s = list(self.setting)
        shift = s[s.index("--shift") + 1] if "--shift" in s else "baseline"
        return "notch_only" if shift == "notch" and "only" in s else shift


def experiment_name(run: Run) -> str:
    import run_experiment

    args = run_experiment.build_parser().parse_args(run.args)
    cfg = run_experiment._apply_mcd_config(run_experiment._build_config(args), args)
    if args.research_subset is not None:
        cfg.subset = args.research_subset
    return cfg.default_name()


def all_runs(attack: str) -> list[Run]:
    runs = []
    for model, (datasets, mcd) in COVERAGE.items():
        for dataset in datasets:
            for setting in grid(attack):
                for subset in ((False, True) if mcd else (False,)):  # research subset only with MCD
                    runs.append(Run(model, dataset, subset, tuple(setting)))
    return runs


def _valid_registry_rows() -> set[tuple[str, str, str]]:
    import pandas as pd

    path = PROJECT_ROOT / "experiments_results.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["timestamp_utc", "model", "dataset", "experiment_name", "shift", "params_json"])
    valid = set()
    for r in df.itertuples():
        family = "notch_only" if r.shift == "notch" and '"notch_mode": "only"' in str(r.params_json) else r.shift
        subset_row = "_MCD_" in r.experiment_name or "_research" in r.experiment_name
        if str(r.timestamp_utc) < LEGACY_CUTOFF:
            if family in LEGACY_INVALID or (family in LEGACY_INVALID_FULL and not subset_row):
                continue
        valid.add((r.model, r.dataset, _norm(r.experiment_name)))
    return valid


def _norm(name: str) -> str:
    """Normalise float tokens so legacy names (eps0.1 vs eps0.10) compare equal."""
    import re

    return re.sub(r"\d+\.\d+", lambda m: format(float(m.group(0)), "g"), name)


def pending(attack: str) -> list[Run]:
    done = _valid_registry_rows()
    return [r for r in all_runs(attack) if (r.model, r.dataset, _norm(experiment_name(r))) not in done]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", default="fgsm", choices=["fgsm", "pgd"])
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args(argv)

    runs, todo = all_runs(args.attack), pending(args.attack)
    todo_set = set(todo)
    if args.coverage:
        print(f"Coverage ({args.attack.upper()} grid of {len(grid(args.attack))} settings):")
        for model, (datasets, mcd) in COVERAGE.items():
            for dataset in datasets:
                for subset in ((False, True) if mcd else (False,)):
                    cell = [r for r in runs if (r.model, r.dataset, r.subset) == (model, dataset, subset)]
                    n_todo = sum(r in todo_set for r in cell)
                    mode = ("10-subject MCD" if mcd else "10-subject (no MCD)") if subset else "full eval"
                    print(f"  {model:18s} {dataset:7s} {mode:20s} {len(cell)} settings: "
                          f"{len(cell) - n_todo} already valid, {n_todo} to run")
    print(f"{len(todo)} runs to do")
    if not args.coverage:
        for i, r in enumerate(todo, 1):
            print(f"{i:4d} {r.model:18s} {r.dataset:7s} {'subset' if r.subset else 'full  '} {experiment_name(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
