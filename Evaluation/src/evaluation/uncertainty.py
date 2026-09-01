from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

from epilepsy2bids.annotations import Annotations
from evaluation.constants import FS, to_mask
from evaluation.subset import EvalSubsetConfig, filter_ref_tsvs

MCD_AGREEMENT_COLUMNS = [
    "mcd_mean_agreement",
    "mcd_seizure_agreement",
    "mcd_background_agreement",
    "mcd_fp_agreement",
    "mcd_fn_agreement",
]

MCD_COUNT_COLUMNS = [
    "mcd_n_files",
    "mcd_n_files_seizure",
    "mcd_n_files_background",
    "mcd_n_files_fp",
    "mcd_n_files_fn",
]

MCD_METRIC_COLUMNS = [*MCD_AGREEMENT_COLUMNS, *MCD_COUNT_COLUMNS]

_STRATUM_SPECS = (
    ("mcd_mean_agreement", "mcd_n_files", "mean_agreement"),
    ("mcd_seizure_agreement", "mcd_n_files_seizure", "seizure_agreement"),
    ("mcd_background_agreement", "mcd_n_files_background", "background_agreement"),
    ("mcd_fp_agreement", "mcd_n_files_fp", "fp_agreement"),
    ("mcd_fn_agreement", "mcd_n_files_fn", "fn_agreement"),
)


def _hyp_agreement_path(hyp_tsv: Path) -> Path:
    stem = hyp_tsv.name
    if stem.endswith("_events.tsv"):
        name = stem[: -len("_events.tsv")] + "_agreement.npz"
    else:
        name = hyp_tsv.stem + "_agreement.npz"
    return hyp_tsv.with_name(name)


def align_agreement_to_fs(agreement: np.ndarray, orig_fs: float, target_fs: int = FS) -> np.ndarray:
    if orig_fs == target_fs:
        return agreement.astype(np.float64)
    new_len = int(round(len(agreement) * target_fs / orig_fs))
    if new_len <= 0:
        return np.array([], dtype=np.float64)
    return resample(agreement, new_len).astype(np.float64)


def _stratum_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _nanmean_and_file_count(series: pd.Series) -> tuple[float, int]:
    vals = series.to_numpy(dtype=np.float64)
    valid = ~np.isnan(vals)
    if not valid.any():
        return float("nan"), 0
    return float(np.nanmean(vals)), int(valid.sum())


def _fmt_metric(value: float) -> str:
    return f"{value:.4f}" if not np.isnan(value) else "n/a"


def evaluate_uncertainty(
    ref_folder: str,
    hyp_folder: str,
    output_dir: str | None = None,
    eval_subset: EvalSubsetConfig | None = None,
) -> dict[str, float]:
    """Aggregate Monte Carlo dropout agreement metrics stratified by reference labels.

    When ``eval_subset`` is active, only reference files in that subset are considered
    (same scope as matched SzCORE / MCD inference). Files without agreement NPZ are skipped.
    """
    ref_path = Path(ref_folder)
    hyp_path = Path(hyp_folder)
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    if eval_subset is None:
        eval_subset = EvalSubsetConfig()

    rows = {
        "file": [],
        "mean_agreement": [],
        "seizure_agreement": [],
        "background_agreement": [],
        "fp_agreement": [],
        "fn_agreement": [],
    }

    all_ref_tsvs = sorted(ref_path.glob("sub-*/**/*.tsv"))
    ref_tsvs = filter_ref_tsvs(all_ref_tsvs, ref_path, eval_subset)

    found_any = False
    for ref_tsv in ref_tsvs:
        hyp_tsv = hyp_path / ref_tsv.relative_to(ref_path)
        agreement_path = _hyp_agreement_path(hyp_tsv)
        if not agreement_path.exists():
            continue

        found_any = True
        ref = Annotations.loadTsv(ref_tsv)
        ref_mask = to_mask(ref)

        if hyp_tsv.exists():
            hyp = Annotations.loadTsv(hyp_tsv)
            hyp_mask = to_mask(hyp)
        else:
            hyp_mask = np.zeros_like(ref_mask)

        with np.load(agreement_path) as data:
            agreement = np.asarray(data["agreement"], dtype=np.float64)
            orig_fs = float(data["fs"]) if "fs" in data else float(FS)

        agreement = align_agreement_to_fs(agreement, orig_fs, FS)
        n = min(len(agreement), len(ref_mask), len(hyp_mask))
        agreement = agreement[:n]
        ref_mask = ref_mask[:n]
        hyp_mask = hyp_mask[:n]

        seizure = ref_mask == 1
        background = ref_mask == 0
        fp = (hyp_mask == 1) & (ref_mask == 0)
        fn = (hyp_mask == 0) & (ref_mask == 1)

        rows["file"].append(ref_tsv.name)
        rows["mean_agreement"].append(_stratum_mean(agreement))
        rows["seizure_agreement"].append(_stratum_mean(agreement[seizure]))
        rows["background_agreement"].append(_stratum_mean(agreement[background]))
        rows["fp_agreement"].append(_stratum_mean(agreement[fp]))
        rows["fn_agreement"].append(_stratum_mean(agreement[fn]))

    metrics: dict[str, float] = {key: float("nan") for key in MCD_METRIC_COLUMNS}
    if not found_any:
        print("No MCD agreement files found; uncertainty metrics left empty.")
        return metrics

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "uncertainty.csv", index=False)

    for agreement_key, count_key, column in _STRATUM_SPECS:
        mean_val, file_count = _nanmean_and_file_count(df[column])
        metrics[agreement_key] = mean_val
        metrics[count_key] = float(file_count)

    summary_lines = ["# Monte Carlo dropout agreement"]
    labels = {
        "mcd_mean_agreement": "Mean agreement",
        "mcd_seizure_agreement": "Seizure agreement",
        "mcd_background_agreement": "Background agreement",
        "mcd_fp_agreement": "FP agreement",
        "mcd_fn_agreement": "FN agreement",
    }
    count_keys = {
        "mcd_mean_agreement": "mcd_n_files",
        "mcd_seizure_agreement": "mcd_n_files_seizure",
        "mcd_background_agreement": "mcd_n_files_background",
        "mcd_fp_agreement": "mcd_n_files_fp",
        "mcd_fn_agreement": "mcd_n_files_fn",
    }
    for agreement_key, label in labels.items():
        count = int(metrics[count_keys[agreement_key]])
        summary_lines.append(
            f"- {label:<22}: {_fmt_metric(metrics[agreement_key])}  (n_files={count})"
        )
    summary = "\n".join(summary_lines) + "\n"
    (out_dir / "uncertainty.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return metrics
