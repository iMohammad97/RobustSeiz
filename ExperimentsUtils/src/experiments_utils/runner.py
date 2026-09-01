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
    """Limit inference to an MCD subset (random EDFs or first-N research subjects)."""
    if not _mcd_enabled(env):
        return edfs

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
