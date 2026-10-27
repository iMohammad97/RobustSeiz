# EEGWaveNet (released weights, no training)

EPFL's SzCORE reproduction of Thuwajit et al., "EEGWaveNet: Multiscale CNN-Based Spatiotemporal Feature Extraction for EEG Seizure Detection" (2021): <https://github.com/esl-epfl/eegwavenet_2021>, commit in `UPSTREAM_COMMIT`. Fetched unchanged from upstream (see `README.md`); the upstream LICENSE file is empty, so it is not redistributed here.

- **Model:** multiscale depthwise CNN, 5 cross-validation fold models (45,734 parameters each) voting together.
- **Input:** 18-ch bipolar double banana, 4 s windows at a 1 s stride, 64 Hz low-pass, no normalization.
- **Training data:** CHB-MIT (SzCORE platform card `eegwavenet.yaml`).
- **Benchmark adapter:** `ExperimentsUtils/src/experiments_utils/models/eegwavenet.py` (baseline output identical to upstream `predict()`).

| Dataset | Reported | Why |
|---------|----------|-----|
| TUSZ eval | ✓ | Held-out; unipolar recordings are converted exactly to bipolar. |
| Siena | ✓ | Held-out; converted to bipolar the same way. |
| CHB-MIT | ✗ | Training data. |

- **MCD:** not supported (no dropout layers).
- **Adversarial:** attacks target the ensemble's mean seizure probability; decisions still use the per-fold majority vote.
