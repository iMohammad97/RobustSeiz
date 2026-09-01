# Adding a new model

Copy this checklist when integrating a custom seizure detector.

## 1. Directory layout

```
Model/
└── YourModel/
    └── your_package/
        ├── pyproject.toml
        └── src/your_package/
            ├── __init__.py
            ├── __main__.py      # optional: standalone CLI
            └── detect.py        # core inference
```

## 2. Implement detection

Your adapter function (called from `ExperimentsUtils`) should match:

```python
def detect(eeg, data) -> list[int]:
    """
    eeg:  epilepsy2bids Eeg object (metadata, fs, channels)
    data: (C, T) float array in µV, already shifted by RobustSeiz
          (shifts include band-pass or notch on raw EEG when those environment/noise shifts are used)
    returns: binary mask length T (1 = seizure at sample t)
    """
    ...
```

## 3. Register the model

In `ExperimentsUtils/src/experiments_utils/models/registry.py`:

```python
from experiments_utils.models import your_model

register_model("YourModel", your_model.detect, mcd_fn=your_model.compute_agreement)  # mcd_fn optional
```

## 4. Activate and build

```text
# Model/active_model
YourModel
```

```bash
docker compose -f docker/compose.yaml build --build-arg ACTIVE_MODEL=YourModel
py run_experiment.py --category baseline
```

## 5. I/O contract (required)

| | Format |
|---|--------|
| **Input** | Unipolar EDF, BIDS eval tree under `Data/ConvertedTUSZEval/` |
| **Output** | Mirrored `*_events.tsv` BIDS annotations under `Results/` |
| **Uncertainty (optional)** | Mirrored `*_agreement.npz` when `--enable-mcd` and model supports MCD |
| **Scoring** | SzCORE-style sample + event metrics via `Evaluation/` |

Match SeizureTransformer's output TSV format so evaluation works without changes.

## 6. Adversarial experiments

If your model supports gradient-based attacks, use `experiments_utils.adversarial_predict.forward_with_adversarial()` in your adapter (see `models/seizure_transformer.py`). Otherwise, document that adversarial shifts are unsupported.

## 7. Monte Carlo dropout uncertainty (optional)

Models with dropout layers may optionally register an MCD function for uncertainty quantification. Models **without** dropout can omit this: RobustSeiz still runs normally when `--enable-mcd` is not passed.

```python
import numpy as np

def compute_agreement(eeg, data) -> np.ndarray:
    """
    Return per-sample agreement index φ in [0, 1], same length as detect() mask.
    Use experiments_utils.mcd.enable_dropout_only(model) to keep BatchNorm in eval.
    Read MCD_SAMPLES and MCD_THRESH from os.environ (set by run_experiment.py).
    """
    ...
```

Register in `registry.py`:

```python
register_model("YourModel", your_model.detect, mcd_fn=your_model.compute_agreement)
```

Agreement index per time point:

φ(t) = (1/T) Σ 𝟙{f(x, θ_t) ≥ τ}

Use `experiments_utils.mcd.compute_agreement_index(binary_runs)` on stacked binary predictions.

## 8. Preprocessing vs RobustSeiz shifts

RobustSeiz may apply **notch filtering on raw µV** (`--category noise --shift notch`) before your model runs. If your model has its own notch (e.g. the sample model SeizureTransformer: 1 & 60 Hz after band-pass), compare robustness with and without the RobustSeiz notch shift. Models without internal notch are expected to show larger sensitivity to mismatch mode (mains hum + wrong notch frequency).


## 9. Band-pass vs model preprocessing

RobustSeiz may apply an **alternative band-pass on raw µV** (`--category environment --shift bandpass`) before your model runs. If your model already band-passes internally (e.g. SeizureTransformer: 0.5–120 Hz), compare robustness with and without the RobustSeiz shift: stacked filtering may or may not matter depending on the model.
