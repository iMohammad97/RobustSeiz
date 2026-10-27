"""Standalone EDF -> TSV inference (no benchmark shifts)."""

import torch
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg

from eegnet_seizure.model import load
from eegnet_seizure.predict import dataloader, eventize, per_second, to_samples, window_probs
from eegnet_seizure.preprocessing import preprocess


def main(edf_file, out_file):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load(device)
    x = preprocess(eeg.data, eeg.fs)
    loader, starts = dataloader(x)
    sec = per_second(window_probs(model, loader, device), starts, x.shape[1])
    decisions = eventize(sec, config["threshold"], config["close_sec"], config["min_event_sec"])
    mask = to_samples(decisions, eeg.data.shape[1], eeg.fs)
    Annotations.loadMask(mask.tolist(), eeg.fs).saveTsv(out_file)
