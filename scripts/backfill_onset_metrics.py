#!/usr/bin/env python3
"""Backfill lead/lag onset metrics into experiments_results.csv from existing Results/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _src in ("ExperimentsUtils/src", "Evaluation/src"):
    _p = PROJECT_ROOT / _src
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluation.evaluate import evaluate  # noqa: E402
from evaluation.onset import ONSET_METRIC_COLUMNS  # noqa: E402
from evaluation.registry import RECORD_COLUMNS, update_experiment_onset_metrics  # noqa: E402
from evaluation.subset import EvalSubsetConfig  # noqa: E402


def _parse_params(params_json: str) -> dict:
    if not params_json or str(params_json) == "nan":
        return {}
    return json.loads(params_json)


def _scoring_scopes_from_params(params_json: str) -> tuple[EvalSubsetConfig, EvalSubsetConfig | None]:
    """Return (eval_subset for SzCORE, onset_subset or None to inherit).

    - MCD / explicit eval_subset → SzCORE matched to that subset; onset inherits.
    - Explicit onset_subset only (legacy non-MCD) → SzCORE full; onset limited.
    """
    params = _parse_params(params_json)

    eval_subset = EvalSubsetConfig()
    if params.get("eval_subset") is not None:
        eval_subset = EvalSubsetConfig(
            subset=int(params["eval_subset"]),
            subset_mode=str(params.get("eval_subset_mode", "random")),
            subset_seed=int(params.get("eval_subset_seed", 42)),
        )
    elif params.get("mcd_subset") is not None:
        eval_subset = EvalSubsetConfig(
            subset=int(params["mcd_subset"]),
            subset_mode=str(params.get("mcd_subset_mode", "random")),
            subset_seed=int(params.get("mcd_subset_seed", 42)),
        )

    onset_subset = None
    if "onset_subset" in params:
        onset_subset = EvalSubsetConfig(
            subset=int(params["onset_subset"]),
            subset_mode=str(params.get("onset_subset_mode", "random")),
            subset_seed=int(params.get("onset_subset_seed", 42)),
        )
    return eval_subset, onset_subset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "experiments_results.csv"),
        help="Path to experiments_results.csv",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip rows that already have lead_mean_steps populated.",
    )
    args = parser.parse_args(argv)

    registry_path = Path(args.registry).resolve()
    if not registry_path.exists():
        print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
        return 1

    import pandas as pd

    df = pd.read_csv(registry_path)
    for col in RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    updated = 0
    skipped = 0
    for idx, row in df.iterrows():
        output_dir = Path(str(row["output_dir"]))
        data_dir = Path(str(row["data_dir"]))
        if args.only_missing and pd.notna(row.get("lead_mean_steps")):
            skipped += 1
            continue
        if not output_dir.exists():
            print(f"SKIP missing output_dir: {output_dir}")
            skipped += 1
            continue
        if not data_dir.exists():
            print(f"SKIP missing data_dir: {data_dir}")
            skipped += 1
            continue

        eval_subset, onset_subset = _scoring_scopes_from_params(str(row.get("params_json", "")))
        print(f"Backfilling onset metrics: {row['experiment_name']} ({output_dir})")
        metrics = evaluate(
            str(data_dir),
            str(output_dir),
            output_dir=str(output_dir),
            eval_subset=eval_subset,
            onset_subset=onset_subset,
        )
        onset_metrics = {key: metrics[key] for key in ONSET_METRIC_COLUMNS}
        update_experiment_onset_metrics(registry_path, str(output_dir), onset_metrics)
        updated += 1

    print(f"Done. updated={updated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
