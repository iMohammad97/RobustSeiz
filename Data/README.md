# Data

Eval recordings are **not bundled**. Download the datasets yourself, convert them to BIDS-EEG with [epilepsy2bids](https://github.com/AI4EEG/epilepsy2bids), and place them here. Every tree is mounted read-only during inference: shifts are applied in memory and the files are never modified.

## Datasets covered

| Dataset | `--dataset` | Folder | Subjects | Recordings | Seizures | Montage after conversion | Source |
|---------|-------------|--------|----------|------------|----------|--------------------------|--------|
| TUSZ (eval split) | `TUSZ` | `Data/ConvertedTUSZEval/` | 43 | 864 | 469 | 19-ch unipolar, common average, 256 Hz | [TUH EEG](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) (licence required) |
| Siena Scalp EEG | `Siena` | `Data/ConvertedSiena/` | 14 | 41 | 47 | 19-ch unipolar, common average, 256 Hz | [PhysioNet](https://doi.org/10.13026/s309-a395) · [BIDS on Zenodo](https://doi.org/10.5281/zenodo.10640762) |
| CHB-MIT | `CHBMIT` | `Data/ConvertedCHBMIT/` | 23 (24 cases) | 686 | 198 | 18-ch bipolar double banana, 256 Hz | [PhysioNet](https://doi.org/10.13026/C2K01R) · [BIDS on Zenodo](https://zenodo.org/records/10259996) |

`--dataset` sets the default data folder; `--data-dir /path/to/bids/root` overrides it.

## Expected layout (BIDS)

```
Data/ConvertedTUSZEval/
├── participants.tsv
└── sub-*/
    └── ses-*/
        └── eeg/
            ├── *_eeg.edf          # recording
            └── *_events.tsv       # reference annotations
```

## Setup

1. **TUSZ:** convert with epilepsy2bids and copy only the `eval` split (`split = eval` in `participants.tsv`) into `Data/ConvertedTUSZEval/`.
2. **Siena and CHB-MIT:** download the PhysioNet releases (check them against their `SHA256SUMS`), then:
   ```bash
   python scripts/convert_datasets.py siena  <raw_root> Data/ConvertedSiena
   python scripts/convert_datasets.py chbmit <raw_root> Data/ConvertedCHBMIT
   ```
   `<raw_root>` is the inner release folder (the one containing `PN*/` or `chb*/`). The BIDS versions on Zenodo can be used instead.
3. **Run:** `python run_experiment.py --dataset Siena --category baseline --runner local`

Known conversion issues:

- **Siena on Windows:** epilepsy2bids' Siena annotation parser fails on every file (`OverflowError` from `time.mktime` on `HH.MM.SS` timestamps). `scripts/convert_datasets.py` patches it to seconds since midnight, which gives the documented 47 seizures.
- **CHB-MIT:** `chb12_27` to `chb12_29` are referential recordings; the converter turns them into the same 18-ch bipolar montage as the rest.

## Datasets not covered

- **SeizeIT1** (KU Leuven): no longer shared ([doi:10.48804/P5Q0OJ](https://doi.org/10.48804/P5Q0OJ): "The SeizeIT1 dataset will no longer be shared upon request due to expiry of the ethical approval.").
- **SeizeIT2** (OpenNeuro, 2025): wearable behind-the-ear EEG rather than the 19-electrode 10-20 scalp layout the SzCORE standard and these models assume.

## Evaluation split rules

| Dataset | Evaluation set | Valid for |
|---------|----------------|-----------|
| TUSZ | Official `eval` split (`train`/`dev` are never scored) | Models trained only on TUSZ `train`/`dev`, or not on TUSZ at all |
| Siena | All 14 subjects (no official split exists) | Models never trained, validated or tuned on any Siena recording |
| CHB-MIT | All 24 cases (no official split exists); `chb21` is a later session of `chb01` (`sub-01/ses-02`), so per-subject results count 23 subjects | Bipolar-capable models never trained, validated or tuned on any CHB-MIT recording |

## Model coverage

Recordings are converted to the model's input montage after shifts are applied. Unipolar trees convert exactly to the bipolar double banana; CHB-MIT can't be converted back to unipolar.

| Model | TUSZ | Siena | CHB-MIT | Reason |
|-------|------|-------|---------|--------|
| SeizureTransformer | ✓ | ✗ | ✗ | Siena: all 14 subjects are in its training data. CHB-MIT: needs 19-ch unipolar input. |
| EEGNet | ✓ | ✓ | ✓ | Trained by us on TUSZ `train`+`dev` only, with bipolar input. |
| EEGWaveNet | ✓ | ✓ | ✗ | CHB-MIT is its training data. |
| EventNet | ✓ | ✓ | ✗ | TUSZ: trained on v2.0.3 `train`+`dev` (maintainers' statement). CHB-MIT: needs 19-ch unipolar input. |

Per-model details: `Model/*/MODEL.md` and `Model/SeizureTransformer/README.md`.

Reference: [sz-validation-framework download guide](https://github.com/esl-epfl/sz-validation-framework/blob/main/1-download.md).
