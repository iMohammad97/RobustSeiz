# ExperimentsUtils

RobustSeiz core: shift definitions, experiment configuration, model registry, and inference pipeline.

Models under `Model/` contain **only** detection code. Everything here is shared across all models.

## Package layout

| Module | Role |
|--------|------|
| `shifts.py` | Environment and noise perturbations |
| `lowfreq_shifts.py` | Impedance (B1) and baseline drift (B2) |
| `adversarial.py` | FGSM / PGD attack steps |
| `adversarial_predict.py` | Adversarial forward pass wrapper |
| `apply.py` | Apply shifts from env vars |
| `config.py` | `ExperimentConfig`: naming, env mapping, CSV rows |
| `inference.py` | **RobustSeiz entry point** (shifts → model → TSV) |
| `runner.py` | Batch inference over eval EDFs: Docker (one container per EDF) or local (in-process, resumable) |
| `datasets.py` | Known eval datasets (`--dataset`): folder and native montage |
| `montage.py` | Converts shifted recordings to the model's input montage (unipolar or bipolar double banana) |
| `models/registry.py` | Model specs (adapter, montage, MCD support) and active model selection |
| `models/seizure_transformer.py` | SeizureTransformer adapter |
| `models/eegnet.py` | EEGNet adapter |
| `models/eegwavenet.py` | EEGWaveNet adapter (5-fold ensemble) |
| `models/eventnet.py` | EventNet adapter |
| `mcd.py` | Model-agnostic MCD helpers (`enable_dropout_only`, agreement index) |

## Inference flow

```
EDF → apply_shifts_from_env() → to model montage → registry.get_model_fn()(eeg, data) → BIDS TSV
                               → [optional] registry.get_mcd_fn()(eeg, data) → agreement NPZ
```

Docker runs: `python3 -m experiments_utils /data/<edf> /output/<tsv>`

## Monte Carlo dropout (optional)

Pass `--enable-mcd` to `run_experiment.py` to quantify uncertainty alongside any shift. Env vars (set automatically from CLI):

| Env | Default | Meaning |
|-----|---------|---------|
| `ENABLE_MCD` | off | `1` when MCD is enabled |
| `MCD_SAMPLES` | 20 | Stochastic forward passes T |
| `MCD_THRESH` | 0.8 | Decision threshold τ for agreement index |
| `MCD_SUBSET` | (all) | Limit inference scope when MCD is enabled |
| `MCD_SUBSET_MODE` | `random` | `random` = N EDFs; `research` = first N `sub-*` dirs (all sessions) |
| `MCD_SUBSET_SEED` | 42 | RNG seed for `random` mode only |

```bash
py run_experiment.py --category baseline --enable-mcd
py run_experiment.py --category noise --shift awgn --snr-db 20 --enable-mcd --mcd-samples 20 --mcd-thresh 0.8
py run_experiment.py --category baseline --enable-mcd --mcd-subset 5 --mcd-subset-seed 42
py run_experiment.py --category baseline --enable-mcd --mcd-subset 5 --mcd-subset-mode research
```

Models without a registered `mcd_fn` log a warning and skip uncertainty output; performance metrics still run.

**Output folders:** MCD runs auto-append a suffix to the experiment name (unless `--name` is set), e.g. `Results/Baseline/Baseline_MCD_s20_research5` vs `Results/Baseline/Baseline`. This keeps full, research-subset, and random-subset MCD runs separate from non-MCD baselines.

## Eval subset (SzCORE + lead/lag)

**Default with MCD subset:** when `--mcd-subset` is set, SzCORE and lead/lag are scored on that **same** inference subset (matched scopes for detection, onset, and φ). Non-MCD runs and MCD without a subset still use **full** SzCORE.

Optional override for onset only:

| Flag | Meaning |
|------|---------|
| `--onset-subset` | Limit lead/lag scoring scope (explicit; does not change SzCORE unless MCD subset is also set) |
| `--onset-subset-mode` | `random` = N reference TSV files; `research` = first N `sub-*` dirs |
| `--onset-subset-seed` | RNG seed for `random` mode (default 42) |

See `Evaluation/README.md`.

```bash
py run_experiment.py --category baseline --enable-mcd --mcd-samples 5 --mcd-subset 10 --mcd-subset-mode research
py run_experiment.py --category baseline --onset-subset 5 --onset-subset-mode research
py run_experiment.py --category baseline --skip-inference
python scripts/rescore_mcd_subset_metrics.py --mcd-samples 5
python scripts/backfill_onset_metrics.py
```


## Install

```bash
pip install -e ExperimentsUtils
```

## Adding a model adapter

See `Model/_template/README.md`. Register your `detect()` function in `models/registry.py`.


## Notch shift (`PERTURB=notch`)

Applied on **raw µV** in `apply.py` before the model runs (before model band-pass).

| Env | Meaning |
|-----|---------|
| `NOTCH_PROFILE` | `off`, `50`, `60`, `50_100`, `60_120` |
| `NOTCH_MODE` | `only`: notch profile only; `mismatch`: add mains (`MAINS_HZ`, `AMP_RATIO`) then notch |
| `NOTCH_Q` | IIR notch Q (default 30) |

```bash
py run_experiment.py --category noise --shift notch --notch-profile 50
py run_experiment.py --category noise --shift notch --notch-mode mismatch --notch-profile 50 --mains-hz 60
```


## Band-pass shift (`SHIFT=bandpass`)

Butterworth band-pass on **raw µV** before the model runs: simulates different EEG acquisition filter settings.

| Env | Meaning |
|-----|---------|
| `BANDPASS_PROFILE` | `0.5_40`, `1_40`, `1_30`, `0.5_25` (lower_upper Hz) |
| `BANDPASS_ORDER` | Filter order (default 3) |

```bash
py run_experiment.py --category environment --shift bandpass --bandpass-profile 0.5_40
py run_experiment.py --category environment --shift bandpass --bandpass-profile 1_30
```


## Impedance noise (`SHIFT=impedance`, B1)

Bands: `0_1`, `0.05_0.5`, `0.1_1`. Amplitude: `--amp-mode ratio|uv`. Channels: `--channel-mode all|one|subset`.

## Baseline drift (`SHIFT=drift`, B2)

`--drift-mode sin|sin_band|random_walk|piecewise`; `--intermittent` for time-limited drift.
