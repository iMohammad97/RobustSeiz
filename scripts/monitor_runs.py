#!/usr/bin/env python3
"""Live dashboard for scripts/run_queue.py.

    python scripts/monitor_runs.py            # refreshes every 3 s (Ctrl+C to quit; runs keep going)
    python scripts/monitor_runs.py --once     # print one snapshot
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = PROJECT_ROOT / "Results" / "logs" / "queue" / "queue.json"
TRAIN_LOG = PROJECT_ROOT / "Results" / "logs" / "train_eegnet.log"


def _load(path: Path) -> dict:
    for _ in range(5):  # the writer may be replacing the file at this instant
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            time.sleep(0.1)
    return {}


def _pid_alive(pid) -> bool:
    """Whether the queue runner process still exists (Windows tasklist; no signals sent)."""
    import subprocess

    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True, timeout=5)
        return str(int(pid)) in out.stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return True  # can't tell; don't raise a false alarm


def _dur(sec: float | None) -> str:
    if sec is None or sec != sec or sec < 0:
        return "-"
    sec = int(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}d {h:02d}h{m:02d}m" if d else f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _bar(fraction: float, width: int = 28) -> ProgressBar:
    return ProgressBar(total=1.0, completed=max(0.0, min(1.0, fraction)), width=width)


def _fraction(entry: dict, status: dict) -> float:
    """Progress of one run: inference by EEG hours, plus a small share for scoring."""
    if entry["state"] == "done":
        return 1.0
    if status.get("state") == "scoring":
        return 0.97
    total = status.get("eeg_sec_total") or 0
    return 0.95 * status.get("eeg_sec_done", 0) / total if total else 0.0


def render(queue: dict) -> Group:
    now = time.time()
    runs = queue.get("runs", [])
    statuses = {r["id"]: _load(Path(r["status_file"])) for r in runs if r["state"] in ("running", "done")}
    frac = {r["id"]: _fraction(r, statuses.get(r["id"], {})) for r in runs}

    total_work = sum(r["est_sec"] for r in runs) or 1
    work_done = sum(r["est_sec"] * frac[r["id"]] for r in runs)
    elapsed = now - queue.get("started", now)
    eta = elapsed * (total_work - work_done) / work_done if work_done > 0 else None
    eeg_total = sum(r["eeg_hours"] for r in runs)
    eeg_done = sum(r["eeg_hours"] * frac[r["id"]] for r in runs)
    counts = defaultdict(int)
    for r in runs:
        counts[r["state"]] += 1

    head = Table.grid(padding=(0, 2))
    head.add_row(Text("Runs", style="bold"), _bar(counts["done"] / max(len(runs), 1), 40),
                 Text(f"{counts['done']}/{len(runs)} done | {counts['running']} running | "
                      f"{counts['pending']} pending | {counts['failed']} failed",
                      style="red" if counts["failed"] else ""))
    head.add_row(Text("Work", style="bold"), _bar(work_done / total_work, 40),
                 Text(f"{100 * work_done / total_work:5.1f} %   EEG processed {eeg_done:,.0f} / {eeg_total:,.0f} h"))
    head.add_row(Text("Time", style="bold"), Text(""),
                 Text(f"elapsed {_dur(elapsed)} | ETA {_dur(eta)}"
                      f"{' (finish ~' + time.strftime('%a %d %b %H:%M', time.localtime(now + eta)) + ')' if eta else ''}"
                      f" | {queue.get('jobs', '?')} jobs x {queue.get('threads', '?')} threads | "
                      f"{str(queue.get('attack', '')).upper()} | queue {queue.get('state', '?')}"))
    if TRAIN_LOG.exists() and "[done]" not in TRAIN_LOG.read_text(encoding="utf-8", errors="ignore")[-2000:]:
        last = TRAIN_LOG.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1:] or [""]
        head.add_row(Text("EEGNet", style="bold yellow"), Text(""),
                     Text(f"training in progress (EEGNet runs wait): {last[0][:90]}", style="yellow"))

    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["dataset"], r["mode"])].append(r)
    by_group = Table(title="Progress by model x dataset x mode", expand=True, title_justify="left")
    for col, kw in (("Model", {}), ("Dataset", {}), ("Mode", {}), ("Progress", {"ratio": 2}),
                    ("Done", {"justify": "right"}), ("Run", {"justify": "right"}),
                    ("Fail", {"justify": "right"}), ("Left (est.)", {"justify": "right"})):
        by_group.add_column(col, **kw)
    for (model, dataset, mode), items in sorted(groups.items()):
        w = sum(i["est_sec"] for i in items)
        wd = sum(i["est_sec"] * frac[i["id"]] for i in items)
        left = (w - wd) * (elapsed / work_done if work_done > 0 else 1) / max(queue.get("jobs", 1), 1)
        n_fail = sum(i["state"] == "failed" for i in items)
        by_group.add_row(model, dataset, "full" if mode == "full" else "10-subj MCD", _bar(wd / w if w else 0, 30),
                         f"{sum(i['state'] == 'done' for i in items)}/{len(items)}",
                         str(sum(i["state"] == "running" for i in items)),
                         Text(str(n_fail), style="red" if n_fail else ""), _dur(left) if wd < w else "done")

    active = Table(title="Running now", expand=True, title_justify="left")
    for col, kw in (("Run", {"ratio": 3}), ("Progress", {"ratio": 2}), ("Files", {"justify": "right"}),
                    ("EEG h", {"justify": "right"}), ("Phase", {}), ("Elapsed", {"justify": "right"}),
                    ("ETA", {"justify": "right"})):
        active.add_column(col, **kw)
    for r in (r for r in runs if r["state"] == "running"):
        st = statuses.get(r["id"], {})
        f = frac[r["id"]]
        run_elapsed = now - r.get("started", now)
        run_eta = run_elapsed * (1 - f) / f if f > 0.01 else r["est_sec"] - run_elapsed
        active.add_row(f"{r['model']} | {r['dataset']} | {r['name']}", _bar(f, 24),
                       f"{st.get('files_done', 0)}/{st.get('files_total', '?')}",
                       f"{st.get('eeg_sec_done', 0) / 3600:,.1f}/{st.get('eeg_sec_total', 0) / 3600:,.1f}",
                       st.get("state", "starting"), _dur(run_elapsed), _dur(run_eta))

    recent = Table(title="Recently finished", expand=True, title_justify="left")
    for col, kw in (("Run", {"ratio": 3}), ("Result", {}), ("Event F1", {"justify": "right"}),
                    ("Sens", {"justify": "right"}), ("FP/24h", {"justify": "right"}), ("Took", {"justify": "right"})):
        recent.add_column(col, **kw)
    finished = sorted((r for r in runs if r["state"] in ("done", "failed")), key=lambda r: -r.get("finished", 0))
    for r in finished[:8]:
        st = statuses.get(r["id"], {})
        fmt = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        recent.add_row(f"{r['model']} | {r['dataset']} | {r['name']}",
                       Text("ok", style="green") if r["state"] == "done" else Text(f"FAILED (rc {r.get('rc')})", style="red"),
                       fmt(st.get("event_f1")), fmt(st.get("event_sensitivity")), fmt(st.get("event_fp_per_day"), 1),
                       _dur(r.get("finished", now) - r.get("started", now)))

    parts = [Panel(head, title="RobustSeiz benchmark queue", border_style="cyan"), by_group, active, recent]
    if queue.get("state") == "running" and not _pid_alive(queue.get("pid")):
        parts.insert(0, Panel(
            "The queue runner is NOT running, so no new runs will start and this view is frozen.\n"
            "Restart it (finished runs are skipped, partial ones resume):\n"
            r"  .venv\Scripts\python scripts\run_queue.py --jobs 4 --threads 4",
            title="QUEUE STOPPED", border_style="red"))
    failed = [r for r in runs if r["state"] == "failed"]
    if failed:
        parts.append(Panel("\n".join(f"{r['id']}  ->  {r['log']}" for r in failed), title="Failed runs (see logs)",
                           border_style="red"))
    return Group(*parts)


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    console = Console()
    if args.once:
        console.print(render(_load(QUEUE_FILE)) if QUEUE_FILE.exists() else "No queue yet.")
        return 0
    with Live(render(_load(QUEUE_FILE)), console=console, refresh_per_second=1, screen=False) as live:
        while True:
            time.sleep(args.interval)
            live.update(render(_load(QUEUE_FILE)))


if __name__ == "__main__":
    raise SystemExit(main())
