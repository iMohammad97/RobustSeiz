from __future__ import annotations

import random
import subprocess
from pathlib import Path

MCD_SUBSET_MODE_RANDOM = "random"
MCD_SUBSET_MODE_RESEARCH = "research"
MCD_SUBSET_MODES = (MCD_SUBSET_MODE_RANDOM, MCD_SUBSET_MODE_RESEARCH)


def _edf_output_path(edf: Path, data_root: Path) -> str:
    """Map BIDS *_eeg.edf inputs to matching *_events.tsv outputs."""
    rel = edf.relative_to(data_root).as_posix()
    lower = rel.lower()
    if lower.endswith("_eeg.edf"):
        return rel[:-8] + "_events.tsv"
    if lower.endswith(".edf"):
        return rel[:-4] + "_events.tsv"
    return rel + "_events.tsv"


def _mcd_enabled(env: dict[str, str]) -> bool:
    return env.get("ENABLE_MCD", "").lower() in ("1", "true", "yes")


def _subject_id(edf: Path, data_root: Path) -> str | None:
    for part in edf.relative_to(data_root).parts:
        if part.startswith("sub-"):
            return part
    return None


def _sorted_subjects(edfs: list[Path], data_root: Path) -> list[str]:
    return sorted({subject for edf in edfs if (subject := _subject_id(edf, data_root))})


def select_edf_subset(edfs: list[Path], data_root: Path, env: dict[str, str]) -> list[Path]:
    """Limit inference to an MCD subset (random EDFs or first-N research subjects),
    or to a research subset (first N subjects) for runs without MCD."""
    if not _mcd_enabled(env):
        research_n = str(env.get("SUBSET", "")).strip()
        if not research_n:
            return edfs
        env = {**env, "MCD_SUBSET": research_n, "MCD_SUBSET_MODE": MCD_SUBSET_MODE_RESEARCH}

    subset_s = str(env.get("MCD_SUBSET", "")).strip()
    if not subset_s:
        return edfs

    subset_n = int(subset_s)
    if subset_n <= 0:
        return edfs

    mode = str(env.get("MCD_SUBSET_MODE", MCD_SUBSET_MODE_RANDOM)).strip().lower()
    if mode not in MCD_SUBSET_MODES:
        raise ValueError(
            f"MCD_SUBSET_MODE must be one of {MCD_SUBSET_MODES}, got {mode!r}"
        )

    if mode == MCD_SUBSET_MODE_RESEARCH:
        subjects = _sorted_subjects(edfs, data_root)
        if subset_n >= len(subjects):
            return edfs
        chosen_subjects = set(subjects[:subset_n])
        chosen = sorted(
            edf for edf in edfs if _subject_id(edf, data_root) in chosen_subjects
        )
        print(
            f"MCD research-subset: {len(chosen_subjects)} subjects, "
            f"{len(chosen)} / {len(edfs)} EDFs"
        )
        print(f"  subjects: {', '.join(sorted(chosen_subjects))}")
        return chosen

    if subset_n >= len(edfs):
        return edfs

    seed = int(env.get("MCD_SUBSET_SEED", "42"))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(edfs, subset_n))
    print(f"MCD random-subset: processing {len(chosen)} / {len(edfs)} EDFs (seed={seed})")
    return chosen


def _outputs_done(output_dir: Path, output_rel: str, env: dict[str, str]) -> bool:
    events = output_dir / output_rel
    if not events.exists():
        return False
    if _mcd_enabled(env):
        from experiments_utils.mcd import agreement_output_path

        return agreement_output_path(events).exists()
    return True


def write_run_status(**fields) -> None:
    """Merge fields into the JSON status file named by RUN_STATUS_FILE (no-op when unset)."""
    import json
    import os
    import time

    path = os.getenv("RUN_STATUS_FILE")
    if not path:
        return
    status_path = Path(path)
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        status = {}
    status.update(fields, updated=time.time())
    tmp = status_path.with_suffix(".tmp")
    # Best effort: on Windows the replace fails while another process (monitor, antivirus)
    # has the file open. Progress reporting must never crash a run, so retry briefly, then skip.
    for _ in range(10):
        try:
            tmp.write_text(json.dumps(status), encoding="utf-8")
            os.replace(tmp, status_path)  # atomic, so monitors never read a half-written file
            return
        except OSError:
            time.sleep(0.2)


def _edf_seconds(edf: Path) -> float:
    import pyedflib

    with pyedflib.EdfReader(str(edf)) as reader:
        return float(reader.getFileDuration())


def run_local_inference(
    data_dir: Path,
    output_dir: Path,
    env: dict[str, str],
    project_root: Path,
    resume: bool = False,
) -> None:
    """In-process inference (no Docker): the model is loaded once and reused for every EDF."""
    import os
    import time

    from experiments_utils.inference import run_inference

    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    if os.getenv("TORCH_THREADS"):
        import torch

        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

    edfs = sorted(data_dir.glob("**/*.edf"))
    if not edfs:
        raise FileNotFoundError(f"No EDF files found under {data_dir}")
    edfs = select_edf_subset(edfs, data_dir, env)
    seconds = [_edf_seconds(e) for e in edfs]
    done_sec = 0.0
    write_run_status(
        state="inference", files_total=len(edfs), files_done=0,
        eeg_sec_total=sum(seconds), eeg_sec_done=0.0, inference_started=time.time(),
    )

    start = time.time()
    for i, (edf, sec) in enumerate(zip(edfs, seconds), 1):
        output_rel = _edf_output_path(edf, data_dir)
        if not (resume and _outputs_done(output_dir, output_rel, env)):
            write_run_status(current=edf.relative_to(data_dir).as_posix())
            t0 = time.time()
            run_inference(str(edf), str(output_dir / output_rel), project_root=project_root)
            print(
                f"[{i}/{len(edfs)}] {edf.relative_to(data_dir).as_posix()} "
                f"{time.time() - t0:.1f}s (elapsed {time.time() - start:.0f}s)"
            )
        done_sec += sec
        write_run_status(files_done=i, eeg_sec_done=done_sec)


def run_docker_inference(
    data_dir: Path,
    output_dir: Path,
    env: dict[str, str],
    project_root: Path,
    service: str = "main",
) -> None:
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compose_file = project_root / "docker" / "compose.yaml"

    edfs = sorted(data_dir.glob("**/*.edf"))
    if not edfs:
        raise FileNotFoundError(f"No EDF files found under {data_dir}")

    edfs = select_edf_subset(edfs, data_dir, env)

    for edf in edfs:
        input_rel = edf.relative_to(data_dir).as_posix()
        output_rel = _edf_output_path(edf, data_dir)
        cmd = [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "run",
            "--rm",
            "--remove-orphans",
            "-v",
            f"{data_dir}:/data:ro",
            "-v",
            f"{output_dir}:/output:rw",
        ]
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(["-e", f"INPUT={input_rel}", "-e", f"OUTPUT={output_rel}", service])
        print(f"Running inference for {input_rel}")
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            raise RuntimeError(f"Docker inference failed for {input_rel} (exit {result.returncode})")
