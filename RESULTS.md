# RobustSeiz benchmark results

444 runs: the 37-setting grid for every model and compatible dataset, in full eval and (for MCD-capable models) on the 10-subject research subset. Every run is a row of `experiments_results.csv`; regenerate the tables with `python scripts/summarize_results.py` (CSV copies in `Results/summary/`). Event-level scores use SzCORE scoring; full eval unless stated.

## 1. Coverage

"37 ✓" = all 37 settings of the grid (baseline + 36 shifts, listed in `scripts/plan_runs.py` and `COMMANDS.md`).

| Model | Size | TUSZ eval | Siena | CHB-MIT | MCD (10-subject) |
|---|---|---|---|---|---|
| SeizureTransformer (2025) | ~44M | 37 ✓ | ✗ trained on Siena | ✗ needs unipolar | 37 ✓ on TUSZ |
| EEGNet (2018), trained by us | 1,649 | 37 ✓ | 37 ✓ | 37 ✓ | 37 ✓ on each dataset |
| EEGWaveNet (2021) | 45.7k × 5 | 37 ✓ | 37 ✓ | ✗ trained on CHB-MIT | ✗ no dropout |
| EventNet (2024), SzCORE weights | ~1.7M | 37 ✓ | 37 ✓ | ✗ needs unipolar | ✗ no dropout |

## 2. Clean baselines (full eval)

| Model | Dataset | Sensitivity | Precision | F1 | FP / 24 h | Detected seizures | Mean lag |
|---|---|---|---|---|---|---|---|
| SeizureTransformer | TUSZ | 0.57 | 0.69 | 0.63 | 16.7 | 195 | 30 s |
| EEGNet | TUSZ | 0.58 | 0.57 | 0.57 | 28.0 | 196 | 36 s |
| EEGNet | Siena | 0.66 | 0.42 | 0.51 | 7.3 | 31 | 25 s |
| EEGNet | CHB-MIT | 0.76 | 0.05 | 0.09 | 73.8 | 151 | 18 s |
| EEGWaveNet | TUSZ | 0.47 | 0.29 | 0.36 | 74.3 | 161 | 40 s |
| EEGWaveNet | Siena | 0.77 | 0.04 | 0.08 | 139.6 | 36 | 35 s |
| EventNet | TUSZ | 0.64 | 0.78 | 0.71 | 11.3 | 219 | 40 s |
| EventNet | Siena | 0.55 | 0.39 | 0.46 | 7.0 | 26 | 12 s |

## 3. Robustness: largest event-F1 drop per shift family (full eval)

Worst setting in each family minus the model's own baseline F1 (negative = worse).

| Family | ST · TUSZ | EEGNet · TUSZ | EEGNet · Siena | EEGNet · CHB-MIT | EEGWaveNet · TUSZ | EEGWaveNet · Siena | EventNet · TUSZ | EventNet · Siena |
|---|---|---|---|---|---|---|---|---|
| FGSM | −0.56 | −0.57 | −0.47 | −0.09 | −0.15 | −0.02 | −0.43 | −0.39 |
| Impedance | −0.52 | −0.52 | −0.37 | 0.00 | −0.08 | −0.02 | −0.33 | −0.06 |
| AWGN | −0.11 | −0.47 | −0.35 | 0.00 | −0.02 | −0.02 | −0.14 | −0.39 |
| Band-pass | −0.18 | −0.08 | −0.26 | −0.04 | +0.03 | +0.04 | −0.02 | +0.13 |
| Impulse | −0.03 | −0.17 | −0.32 | 0.00 | −0.01 | −0.01 | −0.05 | −0.36 |
| Named drop | −0.03 | −0.13 | −0.29 | −0.03 | −0.05 | 0.00 | −0.08 | −0.02 |
| ADC quantization | −0.02 | −0.06 | −0.34 | −0.02 | −0.02 | 0.00 | −0.04 | −0.20 |
| FS mismatch | −0.02 | −0.02 | −0.16 | −0.03 | 0.00 | 0.00 | 0.00 | +0.01 |
| Drift | 0.00 | 0.00 | 0.00 | 0.00 | −0.10 | −0.01 | −0.01 | −0.04 |
| Mains | −0.01 | −0.01 | −0.09 | 0.00 | −0.06 | −0.01 | −0.09 | −0.06 |
| Notch (both) | 0.00 | 0.00 | 0.00 | 0.00 | −0.01 | 0.00 | −0.01 | +0.04 |

FGSM also multiplies false alarms: up to 14.5× baseline for SeizureTransformer, 9.6× for EEGNet on TUSZ, 34× for EEGNet on Siena and 12× / 28× for EventNet on TUSZ / Siena.

## 4. MCD agreement φ (10-subject research subset, baseline)

φ = fraction of the 5 dropout passes voting "seizure".

| Model | Dataset | Mean φ | φ on seizures | φ on false positives | φ on false negatives |
|---|---|---|---|---|---|
| SeizureTransformer | TUSZ | 0.031 | 0.15 | 0.76 | 0.04 |
| EEGNet | TUSZ | 0.008 | 0.14 | 1.00 | 0.10 |
| EEGNet | Siena | 0.005 | 0.25 | 0.95 | 0.13 |
| EEGNet | CHB-MIT | 0.082 | 0.55 | 0.99 | 0.35 |

## Main findings

- **Tiny EEGNet nearly matches SeizureTransformer on TUSZ** (F1 0.57 vs 0.63 with 1.6k vs ~44M parameters) and shares its weak spots: FGSM, impedance noise and AWGN.
- **EventNet has the best clean TUSZ score** (F1 0.71, 11 FP/day) but drops to F1 0.46 on Siena, and Siena is where it is fragile.
  - It takes raw µV with no filtering or normalisation. On Siena, AWGN at ≤ 5 dB SNR and impulses (density ≥ 0.005) raise false alarms 8–23× (F1 0.07–0.15); on TUSZ the same noise costs at most 0.14 F1.
  - An extra 0.5–40 Hz band-pass *improves* it on Siena (F1 0.46 → 0.61, false alarms −40 %), like EEGWaveNet: part of the cross-dataset gap is out-of-band content the model was never trained to ignore.
  - It is the only model hurt by 60 Hz mains on TUSZ (sensitivity 0.64 → 0.49 at amplitude 1.0), as it has no notch or low-pass.
  - Most of its detections start *before* the annotated onset (136 of 219 on TUSZ, 22 of 26 on Siena; the other models lead on 0–31 %). It predicts whole events (centre + duration) rather than thresholding a per-sample score.
- **Cross-dataset generalisation is the dominant failure.** EEGNet on CHB-MIT and EEGWaveNet on Siena keep high sensitivity (0.76–0.77) but raise 74–140 false alarms per day (F1 0.08–0.09). Their small robustness drops are a floor effect, not real robustness.
- **Each model's own preprocessing explains several patterns:**
  - Slow drift only affects EEGWaveNet (−0.10 F1). EventNet also has no high-pass but is barely affected (≤ −0.04).
  - Large slow impedance noise suppresses detections in the models that z-score their input (SeizureTransformer, EEGNet): it inflates the scaling and shrinks the real signal. EventNet (raw µV) is also suppressed on TUSZ (sensitivity 0.64 → 0.24 at 200 µV), so input scaling is not the only cause.
  - An extra band-pass improves EEGWaveNet (F1 slightly up, false alarms down 50–80 %).
- **Notch filtering has no effect on any model,** and mains/notch mismatch barely does either (EventNet's 60 Hz mains drop above is the exception).
- **False alarms are confident:** φ on false positives is 0.76–1.0, so dropout-based uncertainty would not flag them.
- **Caveats:**
  - Siena has only 47 seizures and the TUSZ 10-subject subset only ~16 detected events, so those numbers are noisy.
  - EEGNet on Siena detected no seizure early, so it has no lead values.
  - PGD attacks are implemented but not part of this grid (about 8× the cost of FGSM); `python scripts/plan_runs.py --attack pgd` lists them.
