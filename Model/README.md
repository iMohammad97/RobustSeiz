# Model

Pluggable seizure detection models for **RobustSeiz**.

RobustSeiz (`ExperimentsUtils`) handles shifts, montage conversion, inference orchestration, and evaluation. Each model implements detection on (possibly perturbed) EEG data.

## Active model

Pick a model per run with `--model YourModel`, or set the default in `Model/active_model` (`SeizureTransformer`). Docker images are built for one model: `--build-arg ACTIVE_MODEL=YourModel`.

## Bundled models

| Model | Path | Source | Input montage | MCD | Weights |
|-------|------|--------|---------------|-----|---------|
| SeizureTransformer | `SeizureTransformer/` | [Wu et al., 2025](https://arxiv.org/abs/2504.00336), released weights | 19-ch unipolar | ✓ | Download (below) |
| EEGNet | `EEGNet/` | [Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c), trained by us on TUSZ `train`+`dev` | 18-ch bipolar | ✓ | Included |
| EEGWaveNet | `EEGWaveNet/` | [Thuwajit et al., 2021](https://doi.org/10.1109/TII.2021.3133307), EPFL SzCORE reproduction | 18-ch bipolar | ✗ | Fetch from upstream (`EEGWaveNet/README.md`) |
| EventNet | `EventNet/` | [Seeuws et al., 2024](https://doi.org/10.1109/TBME.2024.3375759), EPFL SzCORE reproduction (MIT) | 19-ch unipolar | ✗ | Included |
| | `_template/` | Checklist for adding a new model | | | |

Per-model preprocessing, training data and dataset coverage: `*/MODEL.md` (SeizureTransformer: `SeizureTransformer/README.md`).

Install the models you want to run:

```bash
pip install -e Model/SeizureTransformer/wu_2025 -e Model/EEGNet/eegnet_seizure -e Model/EventNet/eventnet
pip install -e Model/EEGWaveNet/thuwajit_2021   # after fetching it
```

## Model contract

**Input:** `(eeg, data)` where `data` is `(C, T)` µV **after** RobustSeiz shifts, already converted to the montage declared for the model.

**Output:** Binary seizure mask (`list[int]`, one sample per time point). RobustSeiz writes BIDS `*_events.tsv`.

**Uncertainty (optional):** `compute_agreement(eeg, data) -> np.ndarray` for MCD (`--enable-mcd`).

**Adversarial (optional):** a differentiable forward pass on the model's normalised input windows (see the adapters in `ExperimentsUtils/src/experiments_utils/models/`).

### Add your model

1. Create `Model/YourModel/<package>/` with `pyproject.toml` (see `_template/`)
2. Implement adapter in `ExperimentsUtils/src/experiments_utils/models/your_model.py`
3. Register it in `ExperimentsUtils/src/experiments_utils/models/registry.py` (`ModelSpec(module, montage, mcd)`)
4. Add its held-out datasets to `COVERAGE` in `scripts/plan_runs.py`
5. Run: `python run_experiment.py --model YourModel --category baseline --runner local`

## Weights

SeizureTransformer weights are **not** in git. Download from the [Google Drive folder](https://drive.google.com/drive/folders/17pKhwFc4x1_2zwXTndKawoNKlaXIW-VE?usp=sharing) and place at:

```
Model/SeizureTransformer/wu_2025/src/wu_2025/model.pth
```

Required before running SeizureTransformer or building its Docker image.

## Standalone CLI (no shifts)

```bash
python -m wu_2025 input.edf output_events.tsv
```

RobustSeiz entry (with shifts): `python -m experiments_utils input.edf output_events.tsv`
