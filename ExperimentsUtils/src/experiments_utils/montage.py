"""Montage conversion between the dataset's recording montage and a model's input montage."""

from __future__ import annotations

import numpy as np
from epilepsy2bids.eeg import Eeg

from experiments_utils.shifts import channel_electrodes

UNIPOLAR = "unipolar"
BIPOLAR = "bipolar"


def to_bipolar(data: np.ndarray, labels) -> tuple[np.ndarray, list[str]]:
    """Exact unipolar -> 18-ch double banana (``Eeg.BIPOLAR_DBANANA`` order)."""
    index = {}
    for i, label in enumerate(labels):
        electrodes = channel_electrodes(label)
        if len(electrodes) == 1:
            index[next(iter(electrodes))] = i
    rows = []
    for pair in Eeg.BIPOLAR_DBANANA:
        a, b = (next(iter(channel_electrodes(e))) for e in pair.split("-"))
        if a not in index or b not in index:
            raise ValueError(f"Cannot build {pair}: electrode missing from {list(labels)}")
        rows.append(data[index[a]] - data[index[b]])
    return np.asarray(rows), list(Eeg.BIPOLAR_DBANANA)


def convert_montage(data: np.ndarray, labels, source: str, target: str) -> tuple[np.ndarray, list[str]]:
    if source == target:
        return data, list(labels)
    if source == UNIPOLAR and target == BIPOLAR:
        return to_bipolar(data, labels)
    raise ValueError(
        f"Model needs {target} input but the recording is {source}; "
        "bipolar -> unipolar conversion is not possible."
    )
