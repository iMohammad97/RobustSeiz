# Evaluation

SzCORE-compatible evaluation for **RobustSeiz**: compare hypothesis `*_events.tsv` predictions to reference BIDS annotations.

## Metrics

### SzCORE performance

Per recording and aggregate **sample-level** and **event-level**:

- Sensitivity, Precision, F1, FP/24h

### Lead and lag onset timing

Computed in the **same** `evaluate()` ref/hyp loop (no duplicate I/O). Unit: **timesteps @ 256 Hz**.

For each SzCORE event-level **true positive** seizure:

- **Lead**: early detection (`hyp_onset` before `ref_onset`); reported as positive steps; **higher = earlier**
- **Lag**: late detection (`hyp_onset` after `ref_onset`); reported as positive steps; **higher = later**
- **Ontime**: `delta_steps == 0`; counted in `ontime_n_events` only (excluded from lead and lag)

Aggregate columns: mean, median, p90, max, min for lead and lag; counts `lead_n_events`, `lead_total_n_events`, `lag_n_events`, `lag_total_n_events`, `ontime_n_events`, `tp_n_events`.

Always interpret alongside `event_sensitivity`.

Outputs per experiment: `onset.csv` (per-TP seizure rows), `onset.txt` (summary).

### MCD uncertainty (optional)

With `--enable-mcd` (and a model that supports it):

- `mcd_mean_agreement`, `mcd_seizure_agreement`, `mcd_background_agreement`, `mcd_fp_agreement`, `mcd_fn_agreement`
- `mcd_n_files`, `mcd_n_files_seizure`, `mcd_n_files_background`, `mcd_n_files_fp`, `mcd_n_files_fn`

Uses `timescoring` + `epilepsy2bids` at **256 Hz**.

## Eval subset modes

| Mode | How |
|------|-----|
| Full | Default: all reference files (non-MCD / MCD without `--mcd-subset`) |
| Random | `--mcd-subset N --mcd-subset-mode random` (or `--eval-subset` / `--onset-subset`) |
| Research | `--mcd-subset N --mcd-subset-mode research`: first N `sub-*` subjects |

**Default when MCD uses a subset:** SzCORE, lead/lag, and φ are all scored on that **same** subset (matched to inference). Non-MCD runs still use full SzCORE. Explicit `--onset-subset*` can still limit onset alone; `--eval-subset*` limits SzCORE via `py -m evaluation`.

## Install

```bash
pip install -e Evaluation
```

## CLI

```bash
py -m evaluation <ref_root> <hyp_root> --output-dir Results/Noise/AWGN_20
py -m evaluation <ref> <hyp> --eval-subset 10 --eval-subset-mode research --output-dir Results/...
py run_experiment.py --category baseline --skip-inference
python scripts/backfill_onset_metrics.py
python scripts/rescore_mcd_subset_metrics.py --mcd-samples 5
```

## Registry

`evaluation.registry.append_experiment_result()` appends one row to `experiments_results.csv` with all parameters (`params_json`) and metrics. `update_experiment_onset_metrics()` updates onset columns for an existing row (used by backfill).

## API

```python
from evaluation.evaluate import evaluate
from evaluation.subset import EvalSubsetConfig
from evaluation.uncertainty import evaluate_uncertainty

metrics = evaluate(
    "Data/ConvertedTUSZEval",
    "Results/Baseline/Baseline",
    output_dir="Results/Baseline/Baseline",
    eval_subset=EvalSubsetConfig(),  # full SzCORE; set subset=10, subset_mode="research" to match MCD
)
# metrics keys: sample_sensitivity, event_f1, lead_mean_steps, lag_mean_steps, ...

uncertainty = evaluate_uncertainty("Data/ConvertedTUSZEval", "Results/Baseline/Baseline")
```
