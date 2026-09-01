from pathlib import Path

import numpy as np
import pandas as pd

from epilepsy2bids.annotations import Annotations
from timescoring import annotations, scoring

from evaluation.onset import (
    ONSET_METRIC_COLUMNS,
    collect_onset_rows_for_pair,
    finalize_onset_metrics,
)
from evaluation.subset import EvalSubsetConfig, filter_ref_tsvs

from evaluation.constants import FS, to_mask


def compute_scores(tp, fp, ref_true, duration):
    if ref_true > 0:
        sensitivity = tp / ref_true
    else:
        sensitivity = np.nan

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan

    if np.isnan(sensitivity) or np.isnan(precision):
        f1 = np.nan
    elif (sensitivity + precision) == 0:
        f1 = 0
    else:
        f1 = 2 * sensitivity * precision / (sensitivity + precision)

    fp_rate = fp / (duration / 3600 / 24)
    return sensitivity, precision, f1, fp_rate


def evaluate(
    ref_folder: str,
    hyp_folder: str,
    output_dir: str | None = None,
    onset_subset: EvalSubsetConfig | None = None,
    eval_subset: EvalSubsetConfig | None = None,
) -> dict[str, float]:
    """Score SzCORE + lead/lag over reference annotations.

    Args:
        eval_subset: When active, SzCORE aggregates over this subset only.
            Full eval when inactive / None. MCD subset runs should pass the
            same scope used for inference.
        onset_subset: When active, lead/lag only over this subset. When None
            and ``eval_subset`` is active, onset inherits ``eval_subset``.
            When both are inactive, onset uses the full eval (legacy default).
    """
    ref_path = Path(ref_folder)
    hyp_path = Path(hyp_folder)
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    if eval_subset is None:
        eval_subset = EvalSubsetConfig()
    if onset_subset is None:
        # Match SzCORE scope when caller only set eval_subset (MCD auto-match).
        onset_subset = eval_subset if eval_subset.is_active() else EvalSubsetConfig()

    results = {
        "dataset": [],
        "subject": [],
        "file": [],
        "duration": [],
        "tp_sample": [],
        "fp_sample": [],
        "refTrue_sample": [],
        "tp_event": [],
        "fp_event": [],
        "refTrue_event": [],
    }

    onset_rows: list[dict[str, float | int | str]] = []
    all_ref_tsvs = sorted(ref_path.glob("sub-*/**/*.tsv"))
    szcore_ref_tsvs = filter_ref_tsvs(all_ref_tsvs, ref_path, eval_subset)
    onset_ref_tsvs = set(filter_ref_tsvs(all_ref_tsvs, ref_path, onset_subset))

    print(f"Reference folder: {ref_path}")
    print(f"Hypothesis folder: {hyp_path}")
    if eval_subset.is_active():
        print(
            f"SzCORE eval scope: {eval_subset.subset_mode} subset "
            f"(n={eval_subset.subset}, {len(szcore_ref_tsvs)} files)"
        )
    else:
        print(f"SzCORE eval scope: full ({len(szcore_ref_tsvs)} files)")
    if onset_subset.is_active():
        print(
            f"Onset eval scope: {onset_subset.subset_mode} subset "
            f"(n={onset_subset.subset}, {len(onset_ref_tsvs)} files)"
        )
    else:
        print(f"Onset eval scope: full ({len(onset_ref_tsvs)} files)")

    for ref_tsv in szcore_ref_tsvs:
        print(f"Processing file: {ref_tsv}")
        ref = Annotations.loadTsv(ref_tsv)
        ref_mask = to_mask(ref)
        ref_ann = annotations.Annotation(ref_mask, FS)
        hyp_tsv = hyp_path / ref_tsv.relative_to(ref_path)
        print(f"Hyp file: {hyp_tsv}")
        if hyp_tsv.exists():
            hyp = Annotations.loadTsv(hyp_tsv)
            hyp_mask = to_mask(hyp)
            hyp_ann = annotations.Annotation(hyp_mask, FS)
            hyp_fs = FS
        else:
            hyp_mask = np.zeros_like(ref_mask)
            hyp_ann = annotations.Annotation(hyp_mask, ref_ann.fs)
            hyp_fs = FS

        sample_score = scoring.SampleScoring(ref_ann, hyp_ann)
        event_score = scoring.EventScoring(ref_ann, hyp_ann)

        subject = ref_tsv.name.split("_")[0]
        results["dataset"].append("tuh")
        results["subject"].append(subject)
        results["file"].append(ref_tsv.name)
        results["duration"].append(len(ref_ann.mask) / ref_ann.fs)
        results["tp_sample"].append(sample_score.tp)
        results["fp_sample"].append(sample_score.fp)
        results["refTrue_sample"].append(sample_score.refTrue)
        results["tp_event"].append(event_score.tp)
        results["fp_event"].append(event_score.fp)
        results["refTrue_event"].append(event_score.refTrue)

        onset_rows.extend(
            collect_onset_rows_for_pair(
                ref_mask,
                hyp_mask,
                ref_tsv.name,
                subject,
                hyp_fs=hyp_fs,
            )
            if ref_tsv in onset_ref_tsvs
            else []
        )

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results.csv", index=False)

    metrics: dict[str, float] = {}
    for level, tp_col, fp_col, ref_col in (
        ("sample", "tp_sample", "fp_sample", "refTrue_sample"),
        ("event", "tp_event", "fp_event", "refTrue_event"),
    ):
        sens, prec, f1, fp_rate = compute_scores(
            df[tp_col].sum(),
            df[fp_col].sum(),
            df[ref_col].sum(),
            df["duration"].sum(),
        )
        metrics[f"{level}_sensitivity"] = sens
        metrics[f"{level}_precision"] = prec
        metrics[f"{level}_f1"] = f1
        metrics[f"{level}_fp_per_day"] = fp_rate

    summary = (
        "# Sample scoring\n"
        + f"- Sensitivity : {metrics['sample_sensitivity']:.2f} \n"
        + f"- Precision   : {metrics['sample_precision']:.2f} \n"
        + f"- F1-score    : {metrics['sample_f1']:.2f} \n"
        + f"- FP/24h      : {metrics['sample_fp_per_day']:.2f} \n"
        + "\n# Event scoring\n"
        + f"- Sensitivity : {metrics['event_sensitivity']:.2f} \n"
        + f"- Precision   : {metrics['event_precision']:.2f} \n"
        + f"- F1-score    : {metrics['event_f1']:.2f} \n"
        + f"- FP/24h      : {metrics['event_fp_per_day']:.2f} \n"
    )
    (out_dir / "results.txt").write_text(summary, encoding="utf-8")
    print(summary)

    metrics.update(finalize_onset_metrics(onset_rows, out_dir))
    return metrics


def onset_metric_columns() -> list[str]:
    return list(ONSET_METRIC_COLUMNS)
