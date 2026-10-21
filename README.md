# RobustSeiz

Despite strong predictive performance on held-out electroencephalography (EEG) seizure detection, the clinical utility of seizure detectors remains limited when models face real-world acquisition variability, artifacts, and adversarial inputs. **RobustSeiz** is an open-source, model-agnostic framework for pre-deployment diagnostic reports of detector robustness under controlled, domain-guided perturbations. Plug in your model, apply shifts (environment, noise, adversarial), run GPU inference, and score with SzCORE-style metrics plus optional lead/lag onset timing and Monte Carlo dropout (MCD) uncertainty.

**Sample model (demonstration):** [SeizureTransformer](https://arxiv.org/abs/2504.00336): bundled with demo runs; download weights separately (see below).

## Quick start

```bash
git clone https://github.com/iMohammad97/RobustSeiz.git
cd RobustSeiz

pip install -r requirements.txt
# Download sample-model weights from Google Drive (see Model/README.md) and place at:
# Model/SeizureTransformer/wu_2025/src/wu_2025/model.pth
# Prepare eval data: see Data/README.md
# Build Docker image (GPU required)
docker compose -f docker/compose.yaml build

# Baseline (no shift)
py run_experiment.py --category baseline

# Example shift
py run_experiment.py --category noise --shift awgn --snr-db 20
```

Each run writes predictions under `Results/` and appends one row to **`experiments_results.csv`**.

## Layout

| Path | Purpose |
|------|---------|
| `run_experiment.py` | End-to-end CLI (shift → infer → evaluate → registry) |
| `ExperimentsUtils/` | Shifts, config, model registry, inference pipeline |
| `Evaluation/` | SzCORE scoring, onset timing, MCD uncertainty |
| `Model/` | Pluggable detectors (`Model/active_model` selects which one) |
| `docker/` | GPU inference container |
| `Data/` | Eval dataset (user-provided; see `Data/README.md`) |
| `Results/` | Demo runs + your experiment outputs |
| `experiments_results.csv` | Master metrics log (203 demo runs included) |
| `COMMANDS.md` | Compact command reference for every shift |

## Plug in your model

1. Add package under `Model/YourModel/` (see `Model/_template/README.md`)
2. Implement adapter in `ExperimentsUtils/src/experiments_utils/models/your_model.py`
3. Register in `ExperimentsUtils/src/experiments_utils/models/registry.py`
4. Set `Model/active_model` → `YourModel`
5. Rebuild Docker and run baseline

Details: `Model/README.md`, `Model/_template/README.md`

## Demo results

Bundled under `Results/Baseline/`:
- **Baseline/**: full eval run (no shift)
- **Baseline_MCD_s5_research10/**: same with MCD uncertainty (10 subjects, 5 samples)

`experiments_results.csv` logs metrics from **203 demo runs** (all shift types). Only the Baseline output folders above are included; re-run any shift from `COMMANDS.md` to reproduce others.

## Requirements

- Python 3.10+
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker
- Eval data in BIDS layout: TUSZ eval, Siena or CHB-MIT (`Data/README.md`)

## Docs

- `COMMANDS.md`: all shifts, flags, defaults
- `ExperimentsUtils/README.md`: pipeline internals
- `Evaluation/README.md`: metrics and registry
- `docker/README.md`: container setup
- `Data/README.md`: dataset download and conversion
