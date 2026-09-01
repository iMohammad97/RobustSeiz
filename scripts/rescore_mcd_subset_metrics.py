#!/usr/bin/env python3
"""Re-score MCD subset runs: SzCORE + onset + φ on the matched inference subset (no re-inference)."""

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
from evaluation.registry import METRIC_COLUMNS, RECORD_COLUMNS, update_experiment_metrics  # noqa: E402
from evaluation.subset import EvalSubsetConfig  # noqa: E402
from evaluation.uncertainty import evaluate_uncertainty  # noqa: E402


def _parse_params(raw: str) -> dict:
    if not raw or str(raw) == "nan":
        return {}
    return json.loads(raw)


def _scoring_subset_from_params(params: dict) -> EvalSubsetConfig:
    if params.get("eval_subset") is not None:
        return EvalSubsetConfig(
            subset=int(params["eval_subset"]),
            subset_mode=str(params.get("eval_subset_mode", params.get("mcd_subset_mode", "random"))),
            subset_seed=int(params.get("eval_subset_seed", params.get("mcd_subset_seed", 42))),
        )
    if params.get("mcd_subset") is not None:
        return EvalSubsetConfig(
            subset=int(params["mcd_subset"]),
            subset_mode=str(params.get("mcd_subset_mode", "random")),
            subset_seed=int(params.get("mcd_subset_seed", 42)),
        )
    return EvalSubsetConfig()


def _with_eval_subset_params(params: dict, subset: EvalSubsetConfig) -> str:
    updated = dict(params)
    if subset.is_active():
        updated["eval_subset"] = int(subset.subset)
        updated["eval_subset_mode"] = str(subset.subset_mode)
        updated["eval_subset_seed"] = int(subset.subset_seed)
    return json.dumps(updated, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(PROJECT_ROOT / "experiments_results.csv"),
        help="Path to experiments_results.csv",
    )
    parser.add_argument(
        "--mcd-subset",
        type=int,
        default=10,
        help="Only re-score rows with this mcd_subset (default: 10).",
    )
    parser.add_argument(
        "--mcd-subset-mode",
        default="research",
        choices=["random", "research"],
        help="Only re-score rows with this mcd_subset_mode (default: research).",
    )
    parser.add_argument(
        "--mcd-samples",
        type=int,
        default=None,
        help="Optional filter on mcd_samples (default: any).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching rows without scoring or writing.",
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
    for _, row in df.iterrows():
        params = _parse_params(str(row.get("params_json", "")))
        if not params.get("mcd_enabled"):
            skipped += 1
            continue
        if params.get("mcd_subset") is None:
            skipped += 1
            continue
        if int(params["mcd_subset"]) != int(args.mcd_subset):
            skipped += 1
            continue
        if str(params.get("mcd_subset_mode", "random")) != str(args.mcd_subset_mode):
            skipped += 1
            continue
        if args.mcd_samples is not None and int(params.get("mcd_samples", -1)) != int(args.mcd_samples):
            skipped += 1
            continue

        output_dir = Path(str(row["output_dir"]))
        data_dir = Path(str(row["data_dir"]))
        if not output_dir.exists() or not data_dir.exists():
            print(f"SKIP missing paths: {row['experiment_name']}")
            skipped += 1
            continue

        subset = _scoring_subset_from_params(params)
        print(
            f"Re-scoring ({subset.subset_mode} n={subset.subset}): "
            f"{row['experiment_name']}"
        )
        if args.dry_run:
            updated += 1
            continue

        metrics = evaluate(
            str(data_dir),
            str(output_dir),
            output_dir=str(output_dir),
            eval_subset=subset,
        )
        metrics.update(
            evaluate_uncertainty(
                str(data_dir),
                str(output_dir),
                output_dir=str(output_dir),
                eval_subset=subset,
            )
        )
        metric_payload = {key: metrics[key] for key in METRIC_COLUMNS if key in metrics}
        update_experiment_metrics(
            registry_path,
            str(output_dir),
            metric_payload,
            params_json=_with_eval_subset_params(params, subset),
        )
        updated += 1

    print(f"Done. updated={updated} skipped={skipped} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
