# Results

Experiment outputs are written here and are gitignored.

## Layout

```
Results/<Model>/<Dataset>/<Category>/<Experiment>/
├── sub-*/ses-*/eeg/*_events.tsv    # hypothesis events, mirroring the eval tree
├── sub-*/ses-*/eeg/*_agreement.npz # per-sample MCD agreement (only with --enable-mcd)
├── results.csv / results.txt       # SzCORE sample- and event-based scores
└── onset.csv / onset.txt           # lead/lag onset timing among event true positives
```

Example: `python run_experiment.py --model EEGNet --dataset Siena --category noise --shift awgn --snr-db 20 --runner local` writes `Results/EEGNet/Siena/Noise/AWGN_20/`.

## Registry

`experiments_results.csv` at the repo root is the machine-readable robustness report: one row per run with its parameters (`params_json`) and metrics. It holds the 444 benchmark runs summarised in `RESULTS.md` (metrics only; prediction folders are not bundled). Re-create any row with the matching command from `scripts/plan_runs.py`.

`scripts/summarize_results.py` reads the registry and writes summary tables to `Results/summary/`.
