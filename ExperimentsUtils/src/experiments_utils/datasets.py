"""Evaluation datasets known to the benchmark (see Data/README.md for split rules)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_dir: str  # relative to the project root
    montage: str  # "unipolar" (19-ch common average) or "bipolar" (18-ch double banana)


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec("TUSZ", "Data/ConvertedTUSZEval", "unipolar"),
        DatasetSpec("Siena", "Data/ConvertedSiena", "unipolar"),
        DatasetSpec("CHBMIT", "Data/ConvertedCHBMIT", "bipolar"),
    )
}


def get_dataset(name: str) -> DatasetSpec:
    for key, spec in DATASETS.items():
        if key.lower() == name.lower():
            return spec
    raise ValueError(f"Unknown dataset {name!r}; expected one of {tuple(DATASETS)}")
