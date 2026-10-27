# EventNet (released weights, no training)

Vendored copy of EPFL's SzCORE reproduction of Seeuws et al., "Avoiding Post-Processing With Event-Based Detection in Biomedical Signals" (2024): <https://github.com/esl-epfl/eventnet_2024>, commit in `UPSTREAM_COMMIT`, MIT licence.

- **Code and weights:** `model.pth`, `architecture.py` and `main.py` are byte-identical to both the repo and the published image `ghcr.io/esl-epfl/eventnet_2024:latest` (sha256 from the image's wheel `RECORD`).
- **`utils.py`:** taken from the image, which is the code SzCORE actually ran. It uses `DataLoader(batch_size=1)`. The repo's version (`batch_size=128, n_workers=4`) raises `TypeError` and would pass a 2-D array to `find_peaks`.
- **Model:** 1D U-Net (~1.7M parameters) with event-centre and event-duration heads.
- **Input:** 19-ch unipolar, raw µV, no filtering, resampling or normalisation, 256 Hz.
  - Windows are 120 s, non-overlapping, with 128 samples of context on each side.
  - Recordings shorter than 245 s use shorter windows; anything ≤ 25 s gives no detections.
  - The tail after the last full window is not predicted.
- **Output:** Gaussian-smoothed centre map (σ = 100 samples), then `find_peaks` (height 0.44, distance 60 s). Duration head × 5 min.
- **Training data:** TUSZ v2.0.3 `train` + `dev`, as stated by the maintainers in the SzCORE platform card `eventnet.yaml`. It can't be verified independently: no training code or split list is released. The architecture differs from the paper's seizure model (Neureka 2D U-Net, bipolar, 200 Hz, TUSZ v1.5), so the released weights are a separate SzCORE training run.
- **Benchmark adapter:** `ExperimentsUtils/src/experiments_utils/models/eventnet.py`. It batches windows (the network has no cross-window state); baseline masks are identical to upstream `predict()` (0 mismatching samples on 3 TUSZ and 1 Siena recordings).

| Dataset | Reported | Why |
|---------|----------|-----|
| TUSZ eval | ✓ | Held-out per the maintainers' statement (train + dev only). |
| Siena | ✓ | Not in training data. |
| CHB-MIT | ✗ | Incompatible: needs 19-ch unipolar input (`main.py` asserts it). |

- **MCD:** not supported (the `dropout` argument is never used, so there are no dropout layers).
- **Adversarial:** attacks target the sigmoid event-centre map; decisions use the unchanged peak-finding post-processing.
