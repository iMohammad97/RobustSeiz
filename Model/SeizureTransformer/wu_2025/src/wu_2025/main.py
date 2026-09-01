import os

import numpy as np
import torch
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg

from wu_2025.utils import get_dataloader, load_models, predict


def main(edf_file, outFile):
    eeg = Eeg.loadEdf(edfFile=edf_file)
    assert eeg.montage is Eeg.Montage.UNIPOLAR, "Error: Only unipolar montages are supported."

    channels = getattr(eeg, "channels", None)
    print("eeg.channels:", channels)

    data = eeg.data
    fs = eeg.fs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = eeg.data.shape[1]

    model = load_models(device)
    dataloader = get_dataloader(data, fs, batch_size=512)
    y_predict = predict(model, dataloader, device, seq_len)

    out_dir = os.path.dirname(outFile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    hyp = Annotations.loadMask(y_predict, eeg.fs)
    hyp.saveTsv(outFile)
