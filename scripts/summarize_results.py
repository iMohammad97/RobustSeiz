#!/usr/bin/env python3
"""Summarise valid benchmark results (one registry row per model x dataset x mode x setting).

    python scripts/summarize_results.py              # print tables, write Results/summary/*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from plan_runs import COVERAGE, LEGACY_CUTOFF, LEGACY_INVALID, LEGACY_INVALID_FULL, _norm, all_runs, experiment_name  # noqa: E402

OUT = PROJECT_ROOT / "Results" / "summary"
FAMILY_LABEL = {
    "baseline": "Baseline", "bandpass": "Band-pass", "drop_named": "Named drop", "fs_mismatch": "FS mismatch",
    "quant_adc": "ADC quantization", "impedance": "Impedance", "drift": "Drift", "awgn": "AWGN", "mains": "Mains",
    "impulse": "Impulse", "notch_only": "Notch only", "notch": "Notch mismatch", "fgsm": "FGSM",
}


def valid_results(attack: str = "fgsm") -> pd.DataFrame:
    reg = pd.read_csv(PROJECT_ROOT / "experiments_results.csv")
    reg["norm_name"] = reg.experiment_name.map(_norm)
    rows = []
    for run in all_runs(attack):
        name = experiment_name(run)
        cand = reg[(reg.model == run.model) & (reg.dataset == run.dataset) & (reg.norm_name == _norm(name))]
        legacy = cand.timestamp_utc < LEGACY_CUTOFF
        bad = legacy & ((run.family in LEGACY_INVALID) | ((run.family in LEGACY_INVALID_FULL) & (not run.subset)))
        cand = cand[~bad]
        if cand.empty:
            continue
        row = cand.sort_values("timestamp_utc").iloc[-1].to_dict()
        row.update(mode="MCD-10" if run.subset else "full", family=run.family, setting=name)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = valid_results()
    df.to_csv(OUT / "valid_results.csv", index=False)
    pd.set_option("display.width", 220)

    # 1. Coverage
    cov = df.groupby(["model", "dataset", "mode"]).size().unstack("mode").fillna(0).astype(int)
    print("Coverage (valid settings):\n", cov, "\n")

    # 2. Clean baselines
    base = df[df.family == "baseline"].set_index(["model", "dataset", "mode"])
    cols = ["event_sensitivity", "event_precision", "event_f1", "event_fp_per_day", "sample_f1",
            "tp_n_events", "lead_mean_steps", "lag_mean_steps", "mcd_mean_agreement", "mcd_seizure_agreement",
            "mcd_fp_agreement", "mcd_fn_agreement"]
    base = base[cols]
    base.to_csv(OUT / "baselines.csv")
    print("Baselines:\n", base.round(3), "\n")

    # 3. Worst-case event F1 and max FP/24h per family, relative to each model x dataset baseline (full eval)
    full = df[df["mode"] == "full"]
    b = full[full.family == "baseline"].set_index(["model", "dataset"])
    recs = []
    for (model, dataset), g in full.groupby(["model", "dataset"]):
        f1_0, fp_0 = b.loc[(model, dataset), "event_f1"], b.loc[(model, dataset), "event_fp_per_day"]
        for fam, gf in g[g.family != "baseline"].groupby("family"):
            recs.append({"model": model, "dataset": dataset, "family": FAMILY_LABEL.get(fam, fam),
                         "worst_f1": gf.event_f1.min(), "delta_f1": gf.event_f1.min() - f1_0,
                         "max_fp_per_day": gf.event_fp_per_day.max(), "fp_ratio": gf.event_fp_per_day.max() / fp_0})
    fam = pd.DataFrame(recs)
    fam.to_csv(OUT / "family_worst_case.csv", index=False)
    print("Worst-case event F1 change per family (full eval):\n",
          fam.pivot_table(index="family", columns=["model", "dataset"], values="delta_f1").round(3), "\n")
    print("Max FP/24h ratio vs baseline per family (full eval):\n",
          fam.pivot_table(index="family", columns=["model", "dataset"], values="fp_ratio").round(1), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
