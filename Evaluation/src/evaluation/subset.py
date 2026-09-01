from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

SUBSET_MODE_RANDOM = "random"
SUBSET_MODE_RESEARCH = "research"
SUBSET_MODES = (SUBSET_MODE_RANDOM, SUBSET_MODE_RESEARCH)


@dataclass(frozen=True)
class EvalSubsetConfig:
    """Optional eval-scope filter (full eval when ``subset`` is None)."""

    subset: int | None = None
    subset_mode: str = SUBSET_MODE_RANDOM
    subset_seed: int = 42

    def is_active(self) -> bool:
        return self.subset is not None and self.subset > 0


def _subject_id(ref_tsv: Path, data_root: Path) -> str | None:
    for part in ref_tsv.relative_to(data_root).parts:
        if part.startswith("sub-"):
            return part
    return None


def _sorted_subjects(ref_tsvs: list[Path], data_root: Path) -> list[str]:
    return sorted({subject for tsv in ref_tsvs if (subject := _subject_id(tsv, data_root))})


def filter_ref_tsvs(
    ref_tsvs: list[Path],
    data_root: Path,
    config: EvalSubsetConfig,
) -> list[Path]:
    """Limit evaluation to a subset of reference files (mirrors MCD inference subset modes)."""
    if not config.is_active():
        return ref_tsvs

    subset_n = int(config.subset)
    mode = str(config.subset_mode).strip().lower()
    if mode not in SUBSET_MODES:
        raise ValueError(f"subset_mode must be one of {SUBSET_MODES}, got {mode!r}")

    sorted_refs = sorted(ref_tsvs)
    if mode == SUBSET_MODE_RESEARCH:
        subjects = _sorted_subjects(sorted_refs, data_root)
        if subset_n >= len(subjects):
            return sorted_refs
        chosen_subjects = set(subjects[:subset_n])
        chosen = [tsv for tsv in sorted_refs if _subject_id(tsv, data_root) in chosen_subjects]
        print(
            f"Eval research-subset: {len(chosen_subjects)} subjects, "
            f"{len(chosen)} / {len(sorted_refs)} reference files"
        )
        print(f"  subjects: {', '.join(sorted(chosen_subjects))}")
        return chosen

    if subset_n >= len(sorted_refs):
        return sorted_refs

    seed = int(config.subset_seed)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(sorted_refs, subset_n))
    print(
        f"Eval random-subset: {len(chosen)} / {len(sorted_refs)} "
        f"reference files (seed={seed})"
    )
    return chosen
