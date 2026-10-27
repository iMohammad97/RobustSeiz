# EEGNet (trained by us)

[EEGNet](https://doi.org/10.1088/1741-2552/aace8c) (Lawhern et al. 2018) via the [braindecode](https://braindecode.org) reference implementation (default F1=8, D=2, dropout 0.25; 1,649 parameters), trained for seizure detection on 18-ch bipolar EEG.

- **Input:** 18-ch bipolar double banana, 0.5–40 Hz band-pass, 128 Hz, per-channel z-score per recording, 4 s windows.
- **Output:** window probabilities at a 1 s stride, averaged per second, thresholded (threshold tuned on validation), gaps ≤ 2 s closed, events < 2 s dropped.
- **Training data:** TUSZ train+dev (a BIDS tree of those splits, here `ConvertedTUSZ/`: `sub-002`–`sub-633`; `sub-634` excluded as it duplicates eval `sub-000`). Seeded 10 % subject-wise validation split for early stopping and threshold tuning. TUSZ eval is never read.
- **Weights:** `eegnet_seizure/src/eegnet_seizure/weights/` (`model.pt`, `config.json` with threshold, split and training history).
- **Train:** `python -m eegnet_seizure.train --data-dir ConvertedTUSZ --exclude sub-634 --cache-dir Data/cache/eegnet`

| Dataset | Reported | Why |
|---------|----------|-----|
| TUSZ eval | ✓ | Held-out (only train+dev used). |
| Siena | ✓ | Held-out; converted exactly to bipolar. |
| CHB-MIT | ✓ | Held-out; natively bipolar. |

- **MCD:** supported (dropout layers enabled for T passes).
- **Adversarial:** supported on the normalized 128 Hz windows.
