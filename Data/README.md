# Data

Eval recordings are **not bundled**. Download and convert TUSZ yourself, then place files here.

## Expected layout (BIDS)

```
Data/ConvertedTUSZEval/
└── sub-*/
    └── ses-*/
        └── eeg/
            ├── *_eeg.edf          # unipolar montage
            └── *_events.tsv       # reference annotations
```

## Setup

1. **Download TUSZ** from [TUH EEG Seizure Corpus](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) (license required).

2. **Convert to BIDS** using [epilepsy2bids](https://github.com/AI4EEG/epilepsy2bids):
   ```bash
   pip install epilepsy2bids
   # follow epilepsy2bids docs to convert TUSZ → BIDS tree
   ```

3. **Copy the eval split** into `Data/ConvertedTUSZEval/` (mirror the BIDS structure above).

4. **Run experiments** (default data path):
   ```bash
   py run_experiment.py --category baseline
   ```

## Notes

- The bundled sample model (SeizureTransformer) requires **unipolar** montage EDFs.
- Reference TSVs are used for evaluation only; they are never modified.
- Shifts are applied in memory at inference: original EDFs stay read-only.
- Override path: `--data-dir /path/to/your/bids/root`

## Other datasets

Besides TUSZ, RobustSeiz covers two other public scalp-EEG corpora. Convert them with [epilepsy2bids](https://github.com/AI4EEG/epilepsy2bids) (or use the linked BIDS versions) and point `--data-dir` at the tree.

| Dataset | Subjects | Duration [h] | Seizures | Montage | Source |
|---------|----------|--------------|----------|---------|--------|
| Siena Scalp EEG | 14 | 128 | 47 | 19-ch unipolar | [PhysioNet](https://doi.org/10.13026/s309-a395) · [BIDS on Zenodo](https://doi.org/10.5281/zenodo.10640762) |
| CHB-MIT | 23 (24 cases) | 982 | 198 | 18-ch bipolar | [PhysioNet](https://doi.org/10.13026/C2K01R) · [BIDS on Zenodo](https://zenodo.org/records/10259996) |

Not covered:
- **SeizeIT1:** no longer shared (its ethical approval expired).
- **SeizeIT2:** wearable behind-the-ear EEG, not the 10-20 scalp layout.

## Evaluation split rules

| Dataset | Evaluation set | Valid for |
|---------|----------------|-----------|
| TUSZ | Official `eval` split (`train`/`dev` are never scored) | Models not trained or tuned on TUSZ `eval` |
| Siena | All 14 subjects (no official split exists) | Models never trained or tuned on Siena |
| CHB-MIT | All 24 cases; `chb21` is a later session of `chb01` (`sub-01/ses-02`) | Bipolar-capable models never trained or tuned on CHB-MIT |

## Model coverage

| Model | TUSZ | Siena | CHB-MIT |
|-------|------|-------|---------|
| SeizureTransformer | ✓ | ✗ trained on Siena | ✗ needs unipolar input |

Reference: [sz-validation-framework download guide](https://github.com/esl-epfl/sz-validation-framework/blob/main/1-download.md).
