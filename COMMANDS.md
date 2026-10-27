# RobustSeiz: commands

All commands run from repo root after `pip install -r requirements.txt` and installing the model packages (`Model/README.md`). Examples omit `--model`, `--dataset` and `--runner`; add e.g. `--model EEGNet --dataset Siena --runner local` to any of them. The Docker runner (default) needs `docker compose -f docker/compose.yaml build --build-arg ACTIVE_MODEL=<model>` first.

## Shared flags (all categories)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Model/active_model` | `SeizureTransformer`, `EEGNet`, `EEGWaveNet`, `EventNet` |
| `--dataset` | `TUSZ` | `TUSZ`, `Siena`, `CHBMIT` (sets the default `--data-dir`) |
| `--data-dir` | the dataset's folder | Reference BIDS root |
| `--runner` | `docker` | `docker` (one container per EDF) or `local` (in-process) |
| `--resume` | off | Local runner: skip recordings already predicted |
| `--output-dir` | auto `Results/<Category>/<Name>/` | Override output path |
| `--name` | auto from parameters | Experiment folder name |
| `--results-registry` | `experiments_results.csv` | Metrics CSV |
| `--skip-inference` | off | Re-score existing outputs |
| `--skip-evaluate` | off | Inference only |
| `--enable-mcd` | off | Monte Carlo dropout uncertainty |
| `--mcd-samples` | `20` | Stochastic forwards per recording |
| `--mcd-thresh` | `0.8` | Agreement threshold τ |
| `--mcd-subset` | full eval | Limit inference scope |
| `--mcd-subset-mode` | `random` | `random` or `research` (first N subjects) |
| `--mcd-subset-seed` | `42` | RNG for random subset |
| `--research-subset` | full eval | Without MCD: run and score only the first N subjects |
| `--onset-subset` | full eval | Limit lead/lag scoring |
| `--onset-subset-mode` | `random` | `random` or `research` |
| `--onset-subset-seed` | `42` | RNG for random onset subset |

---

## Baseline

No shift applied.

```bash
py run_experiment.py --category baseline
py run_experiment.py --category baseline --enable-mcd
py run_experiment.py --category baseline --enable-mcd --mcd-subset 5 --mcd-subset-mode research
```

---

## Environment (`--category environment`)

Applied on raw µV before model preprocessing.

### `car`: common-average reference

Not in the benchmark grid: the converted TUSZ and Siena trees are already common-average referenced, so this is a near no-op there.

```bash
py run_experiment.py --category environment --shift car
```

### `fs_mismatch`: resample round-trip

| Flag | Default | Values |
|------|---------|--------|
| `--target-fs` | *required* | e.g. `50`, `100`, `200`, `230` |

```bash
py run_experiment.py --category environment --shift fs_mismatch --target-fs 100
```

### `drop_named`: zero named channels

| Flag | Default |
|------|---------|
| `--drop-names` | `T7,T8` |

```bash
py run_experiment.py --category environment --shift drop_named --drop-names T7,T8
```

### `drop`: random channel drop

| Flag | Default |
|------|---------|
| `--n-drop` | `1` |
| `--rng-seed` | omit (non-deterministic) |

```bash
py run_experiment.py --category environment --shift drop --n-drop 2 --rng-seed 42
```

### `quant_adc`: ADC quantization

| Flag | Default |
|------|---------|
| `--bits` | `12` |
| `--adc-fs-uv` | omit (per-channel max) |

```bash
py run_experiment.py --category environment --shift quant_adc --bits 12
```

### `quant`: symmetric quantization

| Flag | Default |
|------|---------|
| `--bits` | `12` |

```bash
py run_experiment.py --category environment --shift quant --bits 12
```

### `bandpass`: acquisition filter

| Flag | Default | Values |
|------|---------|--------|
| `--bandpass-profile` | `0.5_40` | `0.5_40`, `1_40`, `1_30`, `0.5_25` |
| `--bandpass-order` | `3` | positive int |

```bash
py run_experiment.py --category environment --shift bandpass --bandpass-profile 0.5_40
```

### `impedance`: slow contact noise

| Flag | Default | Values |
|------|---------|--------|
| `--impedance-band` | `0_1` | `0_1`, `0.05_0.5`, `0.1_1` |
| `--amp-mode` | `ratio` | `ratio`, `uv` |
| `--amp-ratio` | `0.25`* | positive float |
| `--amp-uv` | `50` | `25`, `50`, `100`, `200` |
| `--channel-mode` | `all` | `all`, `one`, `subset` |
| `--n-channels` | `2` | if `subset` |
| `--lowfreq-order` | `2` | filter order |
| `--rng-seed` | omit | integer |

\* impedance maps CLI default `0.03` → `0.25`

```bash
py run_experiment.py --category environment --shift impedance --impedance-band 0_1 --amp-mode uv --amp-uv 50
```

### `drift`: baseline wandering

| Flag | Default | Values |
|------|---------|--------|
| `--drift-mode` | `sin` | `sin`, `sin_band`, `random_walk`, `piecewise` |
| `--drift-hz` | `0.2` | sin mode frequency |
| `--drift-band` | `0.05_0.5` | `0_1`, `0.05_0.5`, `0.1_1` |
| `--amp-mode` | `ratio` | `ratio`, `uv` |
| `--amp-uv` | `50` | `25`, `50`, `100`, `200` |
| `--amp-ratio` | `0.03` | positive float |
| `--channel-mode` | `all` | `all`, `one`, `subset` |
| `--n-channels` | `2` | if `subset` |
| `--intermittent` | off | flag |
| `--intermittent-duty` | `0.3` | fraction active |
| `--piecewise-segments` | `8` | piecewise mode |
| `--lowfreq-order` | `2` | filter order |
| `--rng-seed` | omit | integer |

```bash
py run_experiment.py --category environment --shift drift --drift-mode sin_band --drift-band 0.05_0.5
py run_experiment.py --category environment --shift drift --drift-mode random_walk --intermittent
```

---

## Noise (`--category noise`)

### `awgn`: additive white Gaussian noise

| Flag | Default |
|------|---------|
| `--snr-db` | `20` |

```bash
py run_experiment.py --category noise --shift awgn --snr-db 20
```

### `mains`: power-line hum

| Flag | Default |
|------|---------|
| `--mains-hz` | `60` |
| `--amp-ratio` | `0.03` |

```bash
py run_experiment.py --category noise --shift mains --mains-hz 60 --amp-ratio 0.5
```

### `impulse`: sparse spikes

| Flag | Default |
|------|---------|
| `--imp-density` | `0.001` |
| `--imp-amp` | `3.0` |
| `--imp-burst` | `1` |
| `--imp-sym` | `1` |
| `--rng-seed` | omit |
| `--debug-pert` | off |

```bash
py run_experiment.py --category noise --shift impulse --imp-density 0.001 --imp-amp 4.0
```

### `notch`: notch filter (or mains mismatch)

| Flag | Default | Values |
|------|---------|--------|
| `--notch-profile` | `off` | `off`, `50`, `60`, `50_100`, `60_120` |
| `--notch-mode` | `only` | `only`, `mismatch` |
| `--notch-q` | `30` | Q factor |
| `--mains-hz` | `60` | for `mismatch` mode |
| `--amp-ratio` | `0.03` | for `mismatch` mode |

```bash
py run_experiment.py --category noise --shift notch --notch-profile 50
py run_experiment.py --category noise --shift notch --notch-mode mismatch --notch-profile 50 --mains-hz 60
```

---

## Adversarial (`--category adversarial`)

Supported by all bundled adapters. Attacks are white-box signed-gradient steps on the model's own (normalised) input windows; `--eps` is relative to the per-window, per-channel standard deviation of that input, so it means the same for every model. Event detection after the attack uses the model's usual threshold and post-processing.

| Flag | Default | Values |
|------|---------|--------|
| `--attack-mode` | `increase` | `increase`, `decrease`, `flip` |
| `--eps` | `0.1` | perturbation budget |
| `--thresh` | `0.5` | flip-target threshold (`flip` mode only) |
| `--alpha` | `0.01` | PGD step size (same scale as `--eps`) |
| `--steps` | `10` | PGD iterations |

```bash
py run_experiment.py --category adversarial --shift fgsm --attack-mode decrease --eps 0.1 --thresh 0.5
py run_experiment.py --category adversarial --shift pgd --attack-mode decrease --eps 0.1 --alpha 0.025 --steps 10
```

---

## Utility commands

```bash
# Re-score without re-running inference
py run_experiment.py --category baseline --skip-inference

# Standalone evaluation
py -m evaluation Data/ConvertedTUSZEval Results/SeizureTransformer/TUSZ/Baseline/Baseline --output-dir Results/SeizureTransformer/TUSZ/Baseline/Baseline

# Backfill onset columns in registry
python scripts/backfill_onset_metrics.py

# Re-score MCD subset metrics
python scripts/rescore_mcd_subset_metrics.py --mcd-samples 5

# Batch grid (bash)
bash scripts/run_new_shifts.sh
```

---

## Benchmark plan

`scripts/plan_runs.py` crosses the 37-setting grid with every model and its held-out datasets (`COVERAGE`), in full eval and, for MCD-capable models, on the first-10-subject research subset with MCD (T=5, τ=0.8). Runs already in `experiments_results.csv` are skipped.

| Model | Datasets | Modes |
|-------|----------|-------|
| SeizureTransformer | TUSZ | full + 10-subject MCD |
| EEGNet | TUSZ, Siena, CHB-MIT | full + 10-subject MCD |
| EEGWaveNet | TUSZ, Siena | full only (no MCD) |
| EventNet | TUSZ, Siena | full only (no MCD) |

Grid: baseline; band-pass 0.5–40 / 1–40 / 0.5–25 Hz; named drop T7,T8 / O1,O2; FS mismatch 50/100/200 Hz; ADC 5 / 10 (100 µV) / 12 bits; impedance 0–1 Hz at 50/100/200 µV; drift sine 0.2 Hz 50 µV, random walk, piecewise; AWGN 0/2/5/20/50 dB; mains 60 Hz × 0.1/0.5/1.0; impulse density 0.0005/0.005/0.01 at 4×RMS; notch-only 50 Hz; 60 Hz mains + 50 Hz notch; FGSM increase/decrease at ε 0.01/0.05/0.1. Seeded shifts use `--rng-seed 42`.

```bash
python scripts/plan_runs.py --coverage             # coverage matrix
python scripts/plan_runs.py                        # list pending runs
python scripts/run_queue.py --jobs 4 --threads 4   # run them in parallel (local runner, resumable)
python scripts/monitor_runs.py                     # live dashboard (Ctrl+C leaves runs going)
python scripts/summarize_results.py                # coverage, baselines, worst case per family
```

Stop the queue gracefully by creating `Results/logs/queue/STOP` (running jobs finish, no new ones start). Per-run logs: `Results/logs/queue/logs/`.

PGD is implemented but left out of the grid (about 8× the cost of FGSM). The PGD plan mirrors the FGSM runs with 10 steps and step size α = 2.5·ε/10:

```bash
python scripts/plan_runs.py --attack pgd
python scripts/run_queue.py --attack pgd --jobs 4 --threads 4
```
