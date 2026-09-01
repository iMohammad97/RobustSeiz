# RobustSeiz: commands

All commands run from repo root after `pip install -r requirements.txt` and `docker compose -f docker/compose.yaml build`.

Eval data default: `--data-dir Data/ConvertedTUSZEval`

## Shared flags (all categories)

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `Data/ConvertedTUSZEval` | Reference BIDS root |
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

Requires model with gradient support (sample SeizureTransformer adapter included).

| Flag | Default | Values |
|------|---------|--------|
| `--attack-mode` | `increase` | `increase`, `decrease`, `flip` |
| `--eps` | `0.1` | perturbation budget |
| `--thresh` | `0.5` | decision threshold |
| `--alpha` | `0.01` | PGD step size |
| `--steps` | `10` | PGD iterations |
| `--open-k` | `5` | morphology open |
| `--close-k` | `5` | morphology close |
| `--min-event-sec` | `2.0` | min event duration |

```bash
py run_experiment.py --category adversarial --shift fgsm --attack-mode decrease --eps 0.1 --thresh 0.5
py run_experiment.py --category adversarial --shift pgd --attack-mode decrease --eps 0.1 --alpha 0.01 --steps 10
```

---

## Utility commands

```bash
# Re-score without re-running inference
py run_experiment.py --category baseline --skip-inference

# Standalone evaluation
py -m evaluation Data/ConvertedTUSZEval Results/Baseline/Baseline --output-dir Results/Baseline/Baseline

# Backfill onset columns in registry
python scripts/backfill_onset_metrics.py

# Re-score MCD subset metrics
python scripts/rescore_mcd_subset_metrics.py --mcd-samples 5

# Batch grid (bash)
bash scripts/run_new_shifts.sh
```
