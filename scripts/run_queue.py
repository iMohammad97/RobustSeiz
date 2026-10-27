#!/usr/bin/env python3
"""Run the pending benchmark plan in parallel, with state files for scripts/monitor_runs.py.

    python scripts/run_queue.py --jobs 4 --threads 4            # FGSM plan
    python scripts/run_queue.py --attack pgd                     # deferred PGD plan

Stop gracefully: create Results/logs/queue/STOP (running jobs finish, no new ones start).
Re-running the command resumes: finished runs are skipped (registry), partial ones resume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from plan_runs import COVERAGE, Run, experiment_name, pending  # noqa: E402

QUEUE_DIR = PROJECT_ROOT / "Results" / "logs" / "queue"
STATUS_DIR = QUEUE_DIR / "status"
LOG_DIR = QUEUE_DIR / "logs"
STOP_FILE = QUEUE_DIR / "STOP"

# Measured on this machine (single process, s of compute per EEG hour); used for ordering and first ETAs.
RATES = {
    "SeizureTransformer": {"base": 13.8, "mcd": 122.9, "fgsm": 47.9, "pgd": 400.4},
    "EEGNet": {"base": 3.4, "mcd": 9.4, "fgsm": 9.9, "pgd": 77.1},
    "EEGWaveNet": {"base": 5.5, "mcd": 5.5, "fgsm": 16.8, "pgd": 123.1},
    "EventNet": {"base": 1.2, "mcd": 1.2, "fgsm": 5.0, "pgd": 25.0},
}
DATASET_HOURS = {"TUSZ": (127.6, 22.6), "Siena": (141.0, 101.9), "CHBMIT": (982.9, 613.4)}


def estimate(run: Run) -> tuple[float, float]:
    """(EEG hours, estimated seconds) for one run."""
    hours = DATASET_HOURS[run.dataset][1 if run.subset else 0]
    r = RATES[run.model]
    rate = r[run.family] if run.family in ("fgsm", "pgd") else r["base"]
    if run.subset and COVERAGE[run.model][1]:
        rate += r["mcd"] - r["base"]
    return hours, hours * (rate + 0.1) + 20


def eegnet_ready() -> bool:
    """Real (not smoke-test) EEGNet weights exist."""
    cfg = PROJECT_ROOT / "Model/EEGNet/eegnet_seizure/src/eegnet_seizure/weights/config.json"
    try:
        return json.loads(cfg.read_text())["training"]["args"]["max_recordings"] == "None"
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def save_queue(queue: dict) -> None:
    """Best effort: on Windows the replace fails while a reader (e.g. the monitor) has the file
    open. Retry briefly, then skip; the in-memory state is saved again on the next change."""
    tmp = QUEUE_DIR / "queue.tmp"
    for _ in range(25):
        try:
            tmp.write_text(json.dumps(queue, indent=1), encoding="utf-8")
            os.replace(tmp, QUEUE_DIR / "queue.json")
            return
        except OSError:
            time.sleep(0.2)
    print("[warn] could not save queue.json this time (file busy); will retry on next update", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="Torch/BLAS threads per job.")
    parser.add_argument("--attack", default="fgsm", choices=["fgsm", "pgd"])
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=2, help="Re-queue a failed run this many times (it resumes).")
    args = parser.parse_args(argv)

    for d in (STATUS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)

    runs = sorted(pending(args.attack), key=lambda r: -estimate(r)[1])  # longest first
    queue = {"started": time.time(), "jobs": args.jobs, "threads": args.threads, "attack": args.attack,
             "pid": os.getpid(), "state": "running", "runs": []}
    for i, run in enumerate(runs):
        name = experiment_name(run)
        hours, est = estimate(run)
        rid = f"{run.model}__{run.dataset}__{name}"
        queue["runs"].append({
            "id": rid, "model": run.model, "dataset": run.dataset, "mode": "subset" if run.subset else "full",
            "name": name, "family": run.family, "eeg_hours": hours, "est_sec": est, "state": "pending",
            "args": run.args, "log": str(LOG_DIR / f"{rid}.log"), "status_file": str(STATUS_DIR / f"{rid}.json"),
        })
    save_queue(queue)
    print(f"Queue: {len(runs)} runs, {args.jobs} jobs x {args.threads} threads, "
          f"~{sum(r['est_sec'] for r in queue['runs']) / 3600:.0f} h of single-process work")

    env_base = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                "TORCH_THREADS": str(args.threads), "PYTHONWARNINGS": "ignore",
                "PYTHONUNBUFFERED": "1"}
    running: dict[str, subprocess.Popen] = {}
    while True:
        for rid, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            entry = next(r for r in queue["runs"] if r["id"] == rid)
            entry.update(state="done" if rc == 0 else "failed", rc=rc, finished=time.time())
            del running[rid]
            print(f"[{entry['state']}] {rid} ({(entry['finished'] - entry['started']) / 60:.1f} min)", flush=True)
            if rc != 0 and entry.get("attempts", 1) <= args.retries:
                entry.update(state="pending", attempts=entry.get("attempts", 1) + 1)  # resumes where it stopped
                print(f"[retry {entry['attempts'] - 1}/{args.retries}] {rid}", flush=True)
            save_queue(queue)

        stopping = STOP_FILE.exists()
        if not stopping:
            for entry in queue["runs"]:
                if len(running) >= args.jobs:
                    break
                if entry["state"] != "pending" or (entry["model"] == "EEGNet" and not eegnet_ready()):
                    continue
                Path(entry["status_file"]).unlink(missing_ok=True)
                cmd = [sys.executable, "-W", "ignore", "run_experiment.py", "--runner", "local", "--resume", *entry["args"]]
                with open(entry["log"], "w", encoding="utf-8") as log:
                    running[entry["id"]] = subprocess.Popen(
                        cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env={**env_base, "RUN_STATUS_FILE": entry["status_file"]},
                    )
                entry.update(state="running", started=time.time())
                print(f"[start] {entry['id']}", flush=True)
                save_queue(queue)

        waiting = [r for r in queue["runs"] if r["state"] == "pending"]
        if not running and (not waiting or stopping):
            break
        time.sleep(args.poll)

    queue["state"] = "stopped" if STOP_FILE.exists() else "finished"
    queue["finished"] = time.time()
    save_queue(queue)
    failed = [r["id"] for r in queue["runs"] if r["state"] == "failed"]
    print(f"Queue {queue['state']}: {sum(r['state'] == 'done' for r in queue['runs'])} done, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
