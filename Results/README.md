# Results

## Bundled demo runs

These folders show expected output from the bundled sample model ([SeizureTransformer](https://arxiv.org/abs/2504.00336)):

| Folder | Description |
|--------|-------------|
| `Baseline/Baseline/` | Full eval, no shift |
| `Baseline/Baseline_MCD_s5_research10/` | MCD uncertainty (10 subjects, 5 dropout samples) |

Each contains mirrored `*_events.tsv` predictions, `results.csv`, `onset.csv`, and optional `*_agreement.npz`.

## Registry

`experiments_results.csv` at repo root logs **203 runs** across all shift types (metrics only). Re-run commands from `COMMANDS.md` to regenerate prediction folders for other shifts.

## New experiments

```bash
py run_experiment.py --category noise --shift awgn --snr-db 20
# → Results/Noise/AWGN_20/
```

Outputs are gitignored except the demo folders above.
