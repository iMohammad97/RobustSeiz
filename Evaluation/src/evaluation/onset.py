from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from timescoring import annotations, scoring

from evaluation.constants import FS
from evaluation.subset import EvalSubsetConfig, filter_ref_tsvs

ONSET_STAT_COLUMNS = ("mean", "median", "p90", "max", "min")

LEAD_METRIC_COLUMNS = [
    "lead_mean_steps",
    "lead_median_steps",
    "lead_p90_steps",
    "lead_max_steps",
    "lead_min_steps",
    "lead_n_events",
    "lead_total_n_events",
]

LAG_METRIC_COLUMNS = [
    "lag_mean_steps",
    "lag_median_steps",
    "lag_p90_steps",
    "lag_max_steps",
    "lag_min_steps",
    "lag_n_events",
    "lag_total_n_events",
]

ONSET_COUNT_COLUMNS = [
    "ontime_n_events",
    "tp_n_events",
]

ONSET_METRIC_COLUMNS = [*LEAD_METRIC_COLUMNS, *LAG_METRIC_COLUMNS, *ONSET_COUNT_COLUMNS]


def _events_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def _first_positive_onset_steps(
    mask: np.ndarray, mask_fs: int, target_fs: int, search_from_step: int
) -> int | None:
    if mask_fs != target_fs:
        ratio = target_fs / mask_fs
        new_len = int(round(len(mask) * ratio))
        if new_len <= 0:
            return None
        indices = np.linspace(0, len(mask) - 1, new_len).astype(int)
        mask = mask[indices]

    search_from_step = max(0, min(search_from_step, len(mask)))
    positives = np.flatnonzero(mask[search_from_step:] == 1)
    if positives.size == 0:
        return None
    return int(search_from_step + positives[0])


def _preprocess_events(
    ann: annotations.Annotation,
    param: scoring.EventScoring.Parameters,
) -> annotations.Annotation:
    merged = scoring.EventScoring._mergeNeighbouringEvents(ann, param.minDurationBetweenEvents)
    return scoring.EventScoring._splitLongEvents(merged, param.maxEventDuration)


def _is_event_tp(
    r_start: float,
    r_end: float,
    hyp_score: annotations.Annotation,
    file_duration: float,
    param: scoring.EventScoring.Parameters,
    scoring_fs: int,
) -> bool:
    ext_start = max(0.0, r_start - param.toleranceStart)
    ext_end = min(file_duration, r_end + param.toleranceEnd)
    if ext_end <= ext_start:
        return False
    i0 = round(ext_start * scoring_fs)
    i1 = round(ext_end * scoring_fs)
    overlap_sec = float(np.sum(hyp_score.mask[i0:i1])) / scoring_fs
    relative_overlap = overlap_sec / (ext_end - ext_start)
    return relative_overlap > param.minOverlap + 1e-6


def _matched_hyp_onset_sec(
    r_start: float,
    r_end: float,
    hyp_proc: annotations.Annotation,
    hyp_mask: np.ndarray,
    hyp_fs: int,
    file_duration: float,
    param: scoring.EventScoring.Parameters,
) -> float | None:
    ext_start = max(0.0, r_start - param.toleranceStart)
    ext_end = min(file_duration, r_end + param.toleranceEnd)

    hyp_onsets = [
        h_start
        for h_start, h_end in hyp_proc.events
        if _events_overlap(h_start, h_end, ext_start, ext_end)
    ]
    if hyp_onsets:
        return float(min(hyp_onsets))

    search_from = round(ext_start * FS)
    hyp_step = _first_positive_onset_steps(hyp_mask, hyp_fs, FS, search_from)
    if hyp_step is None:
        return None
    return hyp_step / FS


def compute_file_onset_rows(
    ref_mask: np.ndarray,
    hyp_mask: np.ndarray,
    hyp_fs: int = FS,
    param: scoring.EventScoring.Parameters | None = None,
) -> list[dict[str, float | int | str]]:
    """Return per-TP-seizure onset rows for one recording."""
    if param is None:
        param = scoring.EventScoring.Parameters()

    ref_ann = annotations.Annotation(ref_mask, FS)
    hyp_ann = annotations.Annotation(hyp_mask, hyp_fs)

    ref_proc = _preprocess_events(ref_ann, param)
    hyp_proc = _preprocess_events(hyp_ann, param)

    scoring_fs = 10
    hyp_score = annotations.Annotation(
        hyp_proc.events,
        scoring_fs,
        round(len(hyp_mask) / hyp_fs * scoring_fs),
    )
    file_duration = len(ref_mask) / FS

    rows: list[dict[str, float | int | str]] = []
    for r_start, r_end in ref_proc.events:
        if not _is_event_tp(r_start, r_end, hyp_score, file_duration, param, scoring_fs):
            continue

        hyp_onset_sec = _matched_hyp_onset_sec(
            r_start, r_end, hyp_proc, hyp_mask, hyp_fs, file_duration, param
        )
        if hyp_onset_sec is None:
            continue

        ref_onset_steps = int(round(r_start * FS))
        hyp_onset_steps = int(round(hyp_onset_sec * FS))
        delta_steps = hyp_onset_steps - ref_onset_steps

        if delta_steps < 0:
            category = "lead"
            lead_steps = -delta_steps
            lag_steps = 0
        elif delta_steps > 0:
            category = "lag"
            lead_steps = 0
            lag_steps = delta_steps
        else:
            category = "ontime"
            lead_steps = 0
            lag_steps = 0

        rows.append(
            {
                "ref_onset_sec": float(r_start),
                "hyp_onset_sec": float(hyp_onset_sec),
                "delta_steps": int(delta_steps),
                "lead_steps": int(lead_steps),
                "lag_steps": int(lag_steps),
                "category": category,
            }
        )
    return rows


def _distribution_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {key: float("nan") for key in ONSET_STAT_COLUMNS}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p90": float(np.nanpercentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def aggregate_onset_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    leads = [int(r["lead_steps"]) for r in rows if int(r["lead_steps"]) > 0]
    lags = [int(r["lag_steps"]) for r in rows if int(r["lag_steps"]) > 0]
    ontime = sum(1 for r in rows if r["category"] == "ontime")
    tp_n = len(rows)

    lead_stats = _distribution_stats(leads)
    lag_stats = _distribution_stats(lags)

    metrics: dict[str, float] = {
        "lead_mean_steps": lead_stats["mean"],
        "lead_median_steps": lead_stats["median"],
        "lead_p90_steps": lead_stats["p90"],
        "lead_max_steps": lead_stats["max"],
        "lead_min_steps": lead_stats["min"],
        "lead_n_events": float(len(leads)),
        "lead_total_n_events": float(len(leads)),
        "lag_mean_steps": lag_stats["mean"],
        "lag_median_steps": lag_stats["median"],
        "lag_p90_steps": lag_stats["p90"],
        "lag_max_steps": lag_stats["max"],
        "lag_min_steps": lag_stats["min"],
        "lag_n_events": float(len(lags)),
        "lag_total_n_events": float(len(lags)),
        "ontime_n_events": float(ontime),
        "tp_n_events": float(tp_n),
    }
    return metrics


def empty_onset_metrics() -> dict[str, float]:
    return {key: float("nan") for key in ONSET_METRIC_COLUMNS}


def _fmt_metric(value: float) -> str:
    return f"{value:.2f}" if not np.isnan(value) else "n/a"


def format_onset_summary(metrics: dict[str, float]) -> str:
    lines = [
        "# Onset timing (lead / lag)",
        (
            f"- Lead mean/median/p90 (steps): "
            f"{_fmt_metric(metrics['lead_mean_steps'])} / "
            f"{_fmt_metric(metrics['lead_median_steps'])} / "
            f"{_fmt_metric(metrics['lead_p90_steps'])}  "
            f"(n={int(metrics['lead_total_n_events'])})"
        ),
        (
            f"- Lead max/min (steps): "
            f"{_fmt_metric(metrics['lead_max_steps'])} / "
            f"{_fmt_metric(metrics['lead_min_steps'])}  "
            f"[higher = earlier detection]"
        ),
        (
            f"- Lag mean/median/p90 (steps): "
            f"{_fmt_metric(metrics['lag_mean_steps'])} / "
            f"{_fmt_metric(metrics['lag_median_steps'])} / "
            f"{_fmt_metric(metrics['lag_p90_steps'])}  "
            f"(n={int(metrics['lag_total_n_events'])})"
        ),
        (
            f"- Lag max/min (steps): "
            f"{_fmt_metric(metrics['lag_max_steps'])} / "
            f"{_fmt_metric(metrics['lag_min_steps'])}  "
            f"[higher = later detection]"
        ),
        (
            f"- Ontime TP seizures: {int(metrics['ontime_n_events'])}  "
            f"| Total TP seizures scored: {int(metrics['tp_n_events'])}"
        ),
        "  Pair with event_sensitivity; lead/lag are conditional on SzCORE event-level TPs.",
    ]
    return "\n".join(lines) + "\n"


def collect_onset_rows_for_pair(
    ref_mask: np.ndarray,
    hyp_mask: np.ndarray,
    file_name: str,
    subject: str,
    hyp_fs: int = FS,
) -> list[dict[str, float | int | str]]:
    file_rows = compute_file_onset_rows(ref_mask, hyp_mask, hyp_fs=hyp_fs)
    for row in file_rows:
        row["file"] = file_name
        row["subject"] = subject
    return file_rows


def finalize_onset_metrics(
    all_rows: list[dict[str, float | int | str]],
    out_dir: Path,
) -> dict[str, float]:
    metrics = aggregate_onset_rows(all_rows)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_dir / "onset.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "file",
                "subject",
                "ref_onset_sec",
                "hyp_onset_sec",
                "delta_steps",
                "lead_steps",
                "lag_steps",
                "category",
            ]
        ).to_csv(out_dir / "onset.csv", index=False)

    summary = format_onset_summary(metrics)
    (out_dir / "onset.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return metrics


def list_reference_tsvs(ref_path: Path, subset: EvalSubsetConfig | None = None) -> list[Path]:
    ref_tsvs = sorted(ref_path.glob("sub-*/**/*.tsv"))
    if subset is None:
        subset = EvalSubsetConfig()
    return filter_ref_tsvs(ref_tsvs, ref_path, subset)
