# Model

Pluggable seizure detection models for **RobustSeiz**.

RobustSeiz (`ExperimentsUtils`) handles shifts, Docker orchestration, and evaluation. Each model implements detection on (possibly perturbed) EEG data.

## Active model

Edit `Model/active_model` (default: `SeizureTransformer` for demonstration). Override at build time: `ACTIVE_MODEL=YourModel`.

## Sample model (demonstration)

| Path | Description |
|------|-------------|
| `SeizureTransformer/` | [SeizureTransformer](https://arxiv.org/abs/2504.00336) (Wu et al., 2025): bundled for end-to-end demo |
| `SeizureTransformer/wu_2025/` | Upstream Python package |
| `_template/` | Checklist for adding a new model |

## Model contract

**Input:** `(eeg, data)` where `data` is `(C, T)` µV **after** RobustSeiz shifts.

**Output:** Binary seizure mask (`list[int]`, one sample per time point). RobustSeiz writes BIDS `*_events.tsv`.

**Uncertainty (optional):** `compute_agreement(eeg, data) -> np.ndarray` for MCD (`--enable-mcd`).

### Add your model

1. Create `Model/YourModel/<package>/` with `pyproject.toml` (see `_template/`)
2. Implement adapter in `ExperimentsUtils/src/experiments_utils/models/your_model.py`
3. Register in `ExperimentsUtils/src/experiments_utils/models/registry.py`
4. Set `Model/active_model` → `YourModel`
5. Rebuild: `docker compose -f docker/compose.yaml build --build-arg ACTIVE_MODEL=YourModel`

## Weights

Sample-model weights are **not** in git. Download from the [Google Drive folder](https://drive.google.com/drive/folders/17pKhwFc4x1_2zwXTndKawoNKlaXIW-VE?usp=sharing) and place at:

```
Model/SeizureTransformer/wu_2025/src/wu_2025/model.pth
```

Required before `docker compose build` or running inference with the sample model.

## Standalone CLI (no shifts)

```bash
python -m wu_2025 input.edf output_events.tsv
```

RobustSeiz entry (with shifts): `python -m experiments_utils input.edf output_events.tsv`
