from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from evaluation.onset import ONSET_METRIC_COLUMNS
from evaluation.uncertainty import MCD_METRIC_COLUMNS

METRIC_COLUMNS = [
    "sample_sensitivity",
    "sample_precision",
    "sample_f1",
    "sample_fp_per_day",
    "event_sensitivity",
    "event_precision",
    "event_f1",
    "event_fp_per_day",
    *ONSET_METRIC_COLUMNS,
    *MCD_METRIC_COLUMNS,
]

RECORD_COLUMNS = [
    "timestamp_utc",
    "model",
    "dataset",
    "experiment_name",
    "category",
    "shift",
    "data_dir",
    "output_dir",
    "params_json",
    *METRIC_COLUMNS,
]


@contextmanager
def registry_lock(csv_path: Path, timeout: float = 600.0, stale_after: float = 300.0):
    """Exclusive lock (lock file) so parallel runs don't overwrite each other's registry rows."""
    lock = Path(str(csv_path) + ".lock")
    start = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_after:
                    lock.unlink(missing_ok=True)  # left behind by a crashed process
                    continue
            except FileNotFoundError:
                continue
            if time.time() - start > timeout:
                raise TimeoutError(f"Registry lock {lock} held for more than {timeout:.0f}s")
            time.sleep(0.2)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def append_experiment_result(csv_path: Path, record: dict) -> None:
    csv_path = Path(csv_path)
    with registry_lock(csv_path):
        _append_experiment_result(csv_path, record)


def _append_experiment_result(csv_path: Path, record: dict) -> None:
    row = {col: record.get(col) for col in RECORD_COLUMNS}
    if csv_path.exists() and csv_path.stat().st_size > 0:
        df = pd.read_csv(csv_path)
        for col in RECORD_COLUMNS:
            if col not in df.columns:
                df[col] = float("nan")
        df = df[RECORD_COLUMNS]
        df = pd.concat([df, pd.DataFrame([row], columns=RECORD_COLUMNS)], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=RECORD_COLUMNS)
    df.to_csv(csv_path, index=False)
    print(f"Appended experiment record to {csv_path}")


def _mask_for_output_dir(df: pd.DataFrame, output_dir: str) -> pd.Series:
    normalized_output = str(Path(output_dir).resolve())
    return df["output_dir"].astype(str).apply(lambda p: str(Path(p).resolve())) == normalized_output


def update_experiment_onset_metrics(csv_path: Path, output_dir: str, onset_metrics: dict[str, float]) -> None:
    """Update onset columns for the row matching ``output_dir``."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    for col in RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    mask = _mask_for_output_dir(df, output_dir)
    if not mask.any():
        raise KeyError(f"No registry row found for output_dir={output_dir!r}")

    for key in ONSET_METRIC_COLUMNS:
        df.loc[mask, key] = onset_metrics.get(key, float("nan"))
    df = df[RECORD_COLUMNS]
    df.to_csv(csv_path, index=False)


def update_experiment_metrics(
    csv_path: Path,
    output_dir: str,
    metrics: dict[str, float],
    params_json: str | None = None,
) -> None:
    """Update metric columns (and optional ``params_json``) for ``output_dir``."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    for col in RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    mask = _mask_for_output_dir(df, output_dir)
    if not mask.any():
        raise KeyError(f"No registry row found for output_dir={output_dir!r}")

    for key in METRIC_COLUMNS:
        if key in metrics:
            df.loc[mask, key] = metrics[key]
    if params_json is not None:
        df.loc[mask, "params_json"] = params_json
    df = df[RECORD_COLUMNS]
    df.to_csv(csv_path, index=False)
