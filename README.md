# RobustSeiz

Despite strong predictive performance on held-out electroencephalography (EEG) seizure detection, the clinical utility of seizure detectors remains limited when models face real-world acquisition variability, artifacts, and adversarial inputs. **RobustSeiz** is an open-source, model-agnostic framework for pre-deployment diagnostic reports of detector robustness under controlled, domain-guided perturbations. Plug in your model, apply shifts (environment, noise, adversarial) in memory, run inference, and score with SzCORE metrics plus lead/lag onset timing and Monte Carlo dropout (MCD) uncertainty.

**Benchmarked models:** SeizureTransformer, EEGNet, EEGWaveNet and EventNet, on the held-out TUSZ eval split, Siena and CHB-MIT. A model is scored on a dataset only when that dataset is held out for it and in a compatible montage (`Data/README.md`). Results: `RESULTS.md`.

## Benchmark at a glance

One row per model × held-out dataset: clean full-eval performance, onset timing, MCD agreement, and robustness over the 36 shifts.

| Model | Dataset | Se | PPV | F1 | FP/24h | Lead (s) | Lag (s) | Onset (s) | φ seizure / FP | Degraded | Worst ΔF1 (shift) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SeizureTransformer | TUSZ | 0.57 | 0.69 | 0.62 | 16.7 | 109 | 30 | +1 | 0.15 / 0.76 | 12/36 | −0.56 (FGSM ↓, ε 0.05) |
| EEGNet | TUSZ | 0.58 | 0.57 | 0.57 | 28.0 | 88 | 36 | +5 | 0.14 / 1.00 | 13/36 | −0.57 (FGSM ↓, ε 0.10) |
| EEGWaveNet | TUSZ | 0.47 | 0.29 | 0.36 | 74.3 | 48 | 40 | +18 | – | 7/36 | −0.14 (FGSM ↓, ε 0.10) |
| EventNet | TUSZ | 0.64 | 0.78 | 0.71 | 11.3 | 80 | 40 | −35 | – | 16/36 | −0.43 (FGSM ↓, ε 0.10) |
| EEGNet | Siena | 0.66 | 0.42 | 0.51 | 7.3 | – | 25 | +25 | 0.25 / 0.95 | 21/36 | −0.47 (FGSM ↓, ε 0.05) |
| EEGWaveNet | Siena | 0.77 | 0.04 | 0.08 | 139.6 | 110 | 35 | −10 | – | 3/36 | −0.02 (FGSM ↑, ε 0.10) |
| EventNet | Siena | 0.55 | 0.39 | 0.46 | 7.0 | 20 | 12 | −15 | – | 10/36 | −0.39 (AWGN 0 dB) |
| EEGNet | CHB-MIT | 0.76 | 0.05 | 0.09 | 73.8 | 71 | 18 | −2 | 0.55 / 0.99 | 7/36 | −0.09 (FGSM ↓, ε 0.10) |

- **Se, PPV, F1, FP/24h:** SzCORE event scoring on the clean full evaluation set.
- **Lead / Lag:** mean seconds before / after the annotated onset, among detected seizures that fired early / late. **Onset:** mean signed delay over all detected seizures (negative = early).
- **φ seizure / FP:** Monte Carlo dropout agreement (5 passes, 10-subject subset) on seizure and on false-positive samples; "–" = model has no dropout. High φ on false positives means false alarms are confident.
- **Degraded:** shifts whose paired subject-bootstrap 95% CI of ΔF1 is below zero with ΔF1 ≤ −0.02. **Worst ΔF1:** largest F1 drop vs clean; ↓ / ↑ = FGSM decrease / increase attack, ε relative to channel s.d.
- Low clean F1 (EEGWaveNet on Siena, EEGNet on CHB-MIT) caps how much F1 can drop, so few degradations there is a floor effect, not robustness. Full tables: `RESULTS.md`.

## Quick start

```bash
git clone https://github.com/iMohammad97/RobustSeiz.git
cd RobustSeiz

pip install -r requirements.txt
pip install -e Model/SeizureTransformer/wu_2025 -e Model/EEGNet/eegnet_seizure -e Model/EventNet/eventnet
# Optional: fetch EEGWaveNet (Model/EEGWaveNet/README.md), then pip install -e Model/EEGWaveNet/thuwajit_2021
# SeizureTransformer weights: see Model/README.md. Eval data: see Data/README.md.

# Baseline (no shift), in-process on CPU or GPU
python run_experiment.py --model EEGNet --dataset Siena --category baseline --runner local

# Example shifts
python run_experiment.py --model EventNet --dataset TUSZ --category noise --shift awgn --snr-db 5 --rng-seed 42 --runner local
python run_experiment.py --model EEGNet --dataset CHBMIT --category adversarial --shift fgsm --attack-mode decrease --eps 0.05 --runner local

# Docker (NVIDIA GPU): one container per recording
docker compose -f docker/compose.yaml build --build-arg ACTIVE_MODEL=EventNet
python run_experiment.py --model EventNet --dataset TUSZ --category baseline
```

Each run writes predictions under `Results/<Model>/<Dataset>/<Category>/<Experiment>/` and appends one row to **`experiments_results.csv`**.

## Full benchmark

The benchmark grid is 37 settings (baseline, 30 environment and noise shifts, 6 FGSM attacks) for every model and compatible dataset, in full-eval mode plus a 10-subject research subset with MCD for MCD-capable models:

```bash
python scripts/plan_runs.py --coverage           # what is done / still to run
python scripts/run_queue.py --jobs 4 --threads 4 # run it in parallel (resumable)
python scripts/monitor_runs.py                   # live dashboard
python scripts/summarize_results.py              # tables in RESULTS.md
```

Details: `COMMANDS.md`.

## Layout

| Path | Purpose |
|------|---------|
| `run_experiment.py` | End-to-end CLI (shift → infer → evaluate → registry) |
| `ExperimentsUtils/` | Shifts, config, datasets, montage conversion, model registry, inference pipeline |
| `Evaluation/` | SzCORE scoring, onset timing, MCD uncertainty |
| `Model/` | Pluggable detectors (`--model` or `Model/active_model` selects one) |
| `scripts/` | Dataset conversion, run plan, parallel queue, monitor, results summary |
| `docker/` | GPU inference container |
| `Data/` | Eval datasets (user-provided; see `Data/README.md`) |
| `Results/` | Your experiment outputs |
| `experiments_results.csv` | Registry of all 444 benchmark runs behind `RESULTS.md` |
| `COMMANDS.md` | Command reference for every shift and mode |

## Plug in your model

1. Add package under `Model/YourModel/` (see `Model/_template/README.md`)
2. Implement adapter in `ExperimentsUtils/src/experiments_utils/models/your_model.py`
3. Register it with its input montage and MCD support in `ExperimentsUtils/src/experiments_utils/models/registry.py`
4. Add it to `COVERAGE` in `scripts/plan_runs.py` with its held-out datasets
5. Run baseline: `python run_experiment.py --model YourModel --category baseline --runner local`

Recordings are converted to the model's montage (19-ch unipolar or 18-ch bipolar double banana) after shifts are applied. Details: `Model/README.md`, `Model/_template/README.md`.

## Requirements

- Python 3.10+ and PyTorch 2.4+
- Optional: Docker with an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`--runner docker`, the default); otherwise use `--runner local`
- Eval data in BIDS layout: TUSZ eval, Siena or CHB-MIT (`Data/README.md`)

## Docs

- `RESULTS.md`: benchmark results and main findings
- `COMMANDS.md`: all shifts, flags, defaults, and the benchmark plan
- `ExperimentsUtils/README.md`: pipeline internals
- `Evaluation/README.md`: metrics and registry
- `Model/README.md`: models, weights, and coverage
- `docker/README.md`: container setup
- `Data/README.md`: dataset download, conversion, and split rules
