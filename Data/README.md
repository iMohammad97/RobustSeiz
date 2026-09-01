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

RobustSeiz expects the same BIDS layout above. Besides TUSZ, three other public scalp-EEG seizure corpora (≥10 subjects, ≥100 h recording) can be converted and used the same way: point `--data-dir` at your converted tree.

| Dataset | Subjects | Duration [h] | Seizures | Source |
|---------|----------|--------------|----------|--------|
| CHB-MIT | 24 | 982 | 198 | [PhysioNet](https://doi.org/10.13026/C2K01R) · [BIDS on Zenodo](https://zenodo.org/records/10259996) |
| Siena Scalp EEG | 14 | 128 | 47 | [PhysioNet](https://doi.org/10.13026/s309-a395) · [BIDS on Zenodo](https://doi.org/10.5281/zenodo.10640762) |
| SeizeIT1 | 42 | 4211 | 182 | [KU Leuven repository](https://doi.org/10.48804/P5Q0OJ) · [download script](https://gist.github.com/danjjl/c94c1ccd9aa0d77409805a11e79ea379) |

Convert raw releases with [epilepsy2bids](https://github.com/AI4EEG/epilepsy2bids), or use the pre-converted BIDS versions where linked. Model adapters may need montage or channel handling adjustments for non-TUSZ data.

Reference: [sz-validation-framework download guide](https://github.com/esl-epfl/sz-validation-framework/blob/main/1-download.md).
