"""Train EEGNet on TUSZ train+dev (18-ch bipolar) with a subject-wise validation split.

Usage:
    python -m eegnet_seizure.train --data-dir ConvertedTUSZ --exclude sub-634 --cache-dir Data/cache/eegnet

TUSZ eval subjects must not be in --data-dir (they are the benchmark test set).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from timescoring import annotations as ts_annotations
from timescoring import scoring

from eegnet_seizure.model import SeizureEEGNet, save
from eegnet_seizure.predict import dataloader, eventize, per_second, window_probs
from eegnet_seizure.preprocessing import MODEL_FS, WINDOW, preprocess, window_starts

CLOSE_SEC = 2
MIN_EVENT_SEC = 2


def _subject(path: Path) -> str:
    return next(p for p in path.parts if p.startswith("sub-"))


def _events_tsv(edf: Path) -> Path:
    return edf.with_name(edf.name[: -len("_eeg.edf")] + "_events.tsv")


def _seizure_mask(tsv: Path, n_samples: int, fs: float) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for event in Annotations.loadTsv(str(tsv)).events:
        if event["eventType"].value != "bckg":
            mask[round(event["onset"] * fs) : round((event["onset"] + event["duration"]) * fs)] = True
    return mask


def _load_bipolar(edf: Path) -> tuple[np.ndarray, float]:
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=str(edf))
    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar()
    return eeg.data, float(eeg.fs)


def _recording_windows(job) -> tuple[str, int, int]:
    """Preprocess one recording and write its sampled windows to a shard."""
    edf, shard, sz_stride, bg_prob, seed = job
    data, fs = _load_bipolar(Path(edf))
    x = preprocess(data, fs)
    sz = _seizure_mask(_events_tsv(Path(edf)), data.shape[1], fs)
    sz = sz[(np.arange(x.shape[1]) * fs / MODEL_FS).astype(int)]  # labels on the 128 Hz grid

    rng = np.random.default_rng(seed)
    windows, labels = [], []
    for s in window_starts(x.shape[1], sz_stride):
        if sz[s : s + WINDOW].mean() >= 0.5:
            windows.append(x[:, s : s + WINDOW]); labels.append(1)
    for s in window_starts(x.shape[1], WINDOW):
        if not sz[s : s + WINDOW].any() and rng.random() < bg_prob:
            windows.append(x[:, s : s + WINDOW]); labels.append(0)
    if windows:
        np.savez(shard, x=np.asarray(windows, dtype=np.float16), y=np.asarray(labels, dtype=np.int8))
    return edf, int(sum(labels)), int(len(labels) - sum(labels))


def _expected_background_prob(edfs: list[Path], sz_stride_sec: float, bg_ratio: float) -> float:
    """Background sampling probability that yields ~bg_ratio x the seizure-window count."""
    sz_sec = bg_sec = 0.0
    for edf in edfs:
        events = Annotations.loadTsv(str(_events_tsv(edf))).events
        dur = float(events[0]["recordingDuration"])
        s = sum(e["duration"] for e in events if e["eventType"].value != "bckg")
        sz_sec += s
        bg_sec += dur - s
    n_sz = sz_sec / sz_stride_sec
    n_bg = bg_sec / (WINDOW / MODEL_FS)
    # Small floor keeps some background when a subset happens to contain no seizures.
    return min(1.0, max(bg_ratio * n_sz / max(n_bg, 1.0), 0.02))


def build_cache(edfs, cache_dir: Path, name: str, args) -> tuple[np.ndarray, np.ndarray]:
    out_x, out_y = cache_dir / f"{name}_x.npy", cache_dir / f"{name}_y.npy"
    if out_x.exists() and out_y.exists():
        print(f"[cache] reusing {out_x}")
        return np.load(out_x, mmap_mode="r"), np.load(out_y)

    shard_dir = cache_dir / f"{name}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    bg_prob = _expected_background_prob(edfs, args.sz_stride_sec, args.bg_ratio)
    print(f"[cache] {name}: {len(edfs)} recordings, background sampling prob {bg_prob:.4f}")
    jobs = [
        (str(e), str(shard_dir / f"{i:05d}.npz"), int(args.sz_stride_sec * MODEL_FS), bg_prob, args.seed + i)
        for i, e in enumerate(edfs)
    ]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (_, n_sz, n_bg) in enumerate(pool.map(_recording_windows, jobs, chunksize=4), 1):
            if k % 200 == 0 or k == len(jobs):
                print(f"[cache] {name}: {k}/{len(jobs)} recordings ({time.time() - t0:.0f}s)")

    shards = sorted(shard_dir.glob("*.npz"))
    counts = []
    for s in shards:
        with np.load(s) as d:  # npz files stay open (and undeletable on Windows) unless closed
            counts.append(len(d["y"]))
    x = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.float16, shape=(sum(counts), 18, WINDOW))
    y = np.zeros(sum(counts), dtype=np.int8)
    i = 0
    for s, n in zip(shards, counts):
        with np.load(s) as d:
            x[i : i + n], y[i : i + n] = d["x"], d["y"]
        i += n
    x.flush()
    np.save(out_y, y)
    for s in shards:
        s.unlink()
    shard_dir.rmdir()
    print(f"[cache] {name}: {len(y)} windows ({int(y.sum())} seizure, {int((y == 0).sum())} background)")
    return np.load(out_x, mmap_mode="r"), y


def _batches(n: int, batch: int, shuffle: bool, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch):
        yield np.sort(idx[i : i + batch])


def train(x_tr, y_tr, x_va, y_va, args) -> tuple[SeizureEEGNet, list[dict]]:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = SeizureEEGNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss()
    x_tr = np.asarray(x_tr)  # load into RAM (float16)
    x_va = np.asarray(x_va)
    history, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for b in _batches(len(y_tr), args.batch_size, True, rng):
            xb = torch.from_numpy(x_tr[b].astype(np.float32))
            yb = torch.from_numpy(y_tr[b].astype(np.float32))
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(b)
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b in _batches(len(y_va), 4096, False, rng):
                p = model(torch.from_numpy(x_va[b].astype(np.float32)))
                va_loss += loss_fn(p, torch.from_numpy(y_va[b].astype(np.float32))).item() * len(b)
        if len(y_va) == 0:
            raise ValueError("Validation split produced no windows")
        rec = {"epoch": epoch, "train_loss": tr_loss / len(y_tr), "val_loss": va_loss / len(y_va), "sec": time.time() - t0}
        history.append(rec)
        print(f"[train] epoch {epoch}: train {rec['train_loss']:.4f} val {rec['val_loss']:.4f} ({rec['sec']:.0f}s)")
        if rec["val_loss"] < best - 1e-4:
            best, bad = rec["val_loss"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train] early stop (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)
    return model.eval(), history


def tune_threshold(model, edfs) -> tuple[float, dict]:
    """Pick the decision threshold maximising SzCORE event F1 on full validation recordings."""
    recs = []
    for edf in edfs:
        data, fs = _load_bipolar(edf)
        x = preprocess(data, fs)
        loader, starts = dataloader(x)
        sec = per_second(window_probs(model, loader, "cpu"), starts, x.shape[1])
        ref = _seizure_mask(_events_tsv(edf), data.shape[1], fs)
        ref_sec = np.array([ref[int(i * fs) : int((i + 1) * fs)].mean() >= 0.5 for i in range(len(sec))])
        recs.append((sec, ref_sec))
    scores = {}
    for thr in np.round(np.arange(0.30, 0.99, 0.05), 2):
        tp = fp = ref_true = 0
        for sec, ref_sec in recs:
            hyp = eventize(sec, float(thr), CLOSE_SEC, MIN_EVENT_SEC).astype(bool)
            s = scoring.EventScoring(ts_annotations.Annotation(ref_sec, 1), ts_annotations.Annotation(hyp, 1))
            tp, fp, ref_true = tp + s.tp, fp + s.fp, ref_true + s.refTrue
        sens = tp / ref_true if ref_true else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * sens * prec / (sens + prec) if sens + prec else 0.0
        scores[float(thr)] = {"event_f1": f1, "event_sensitivity": sens, "event_precision": prec}
        print(f"[tune] thr {thr:.2f}: event F1 {f1:.3f} (sens {sens:.3f}, prec {prec:.3f})")
    best = max(scores, key=lambda t: (scores[t]["event_f1"], t))
    return best, scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[], help="Subjects to leave out (e.g. eval duplicates).")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sz-stride-sec", type=float, default=2.0)
    parser.add_argument("--bg-ratio", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--max-recordings", type=int, default=None, help="Smoke tests only.")
    args = parser.parse_args(argv)

    edfs = sorted(args.data_dir.glob("sub-*/ses-*/eeg/*_eeg.edf"))
    edfs = [e for e in edfs if _subject(e) not in set(args.exclude)]
    subjects = sorted({_subject(e) for e in edfs})
    random.Random(args.seed).shuffle(subjects)
    n_val = max(1, round(len(subjects) * args.val_frac))
    val_subjects = set(subjects[:n_val])
    tr_edfs = [e for e in edfs if _subject(e) not in val_subjects]
    va_edfs = [e for e in edfs if _subject(e) in val_subjects]
    if args.max_recordings:
        tr_edfs, va_edfs = tr_edfs[: args.max_recordings], va_edfs[: max(1, args.max_recordings // 5)]
    print(f"[split] {len(subjects) - n_val} train / {n_val} val subjects; {len(tr_edfs)} / {len(va_edfs)} recordings")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr = build_cache(tr_edfs, args.cache_dir, "train", args)
    x_va, y_va = build_cache(va_edfs, args.cache_dir, "val", args)
    model, history = train(x_tr, y_tr, x_va, y_va, args)
    threshold, threshold_scores = tune_threshold(model, va_edfs)

    config = {
        "threshold": threshold,
        "close_sec": CLOSE_SEC,
        "min_event_sec": MIN_EVENT_SEC,
        "training": {
            "data": "TUSZ train+dev (BIDS), 18-ch bipolar double banana",
            "excluded_subjects": args.exclude,
            "train_subjects": len(subjects) - n_val,
            "val_subjects": sorted(val_subjects),
            "train_windows": {"seizure": int(y_tr.sum()), "background": int((y_tr == 0).sum())},
            "args": {k: str(v) for k, v in vars(args).items()},
            "history": history,
            "threshold_scores": threshold_scores,
            "versions": {"torch": torch.__version__, "braindecode": version("braindecode")},
        },
    }
    save(model, config)
    print(f"[done] threshold {threshold} (val event F1 {threshold_scores[threshold]['event_f1']:.3f}); weights saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
