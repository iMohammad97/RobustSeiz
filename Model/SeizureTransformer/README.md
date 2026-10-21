# SeizureTransformer (sample model)

This folder contains the upstream **SeizureTransformer** package only: no RobustSeiz shifts or experiment code. It is bundled as a **demonstration model** to show how to plug a detector into the framework.

- Paper: [arXiv:2504.00336](https://arxiv.org/abs/2504.00336)
- Weights: see `Model/README.md`

## Dataset coverage

The bundled weights are the authors' 2025 Seizure Detection Challenge model, trained on the TUSZ `train` split plus all of Siena (paper §3.4; `time_step_level/get_dataset.py`). RobustSeiz therefore reports it on **TUSZ eval only**:
- **Siena:** no held-out subjects.
- **CHB-MIT:** the model needs 19-ch unipolar input; CHB-MIT is bipolar and can't be converted back.

## Package layout

```
wu_2025/src/wu_2025/
├── architecture.py   # Network definition
├── utils.py          # Preprocessing, forward pass, eventization
├── main.py           # EDF → TSV (clean inference, no shifts)
└── model.pth         # Weights (download separately; see Model/README.md)
```

## Direct usage

```bash
pip install -e Model/SeizureTransformer/wu_2025
python -m wu_2025 recording.edf recording_events.tsv
```

## In RobustSeiz

Shifts and adversarial attacks are applied by `ExperimentsUtils` before calling `experiments_utils.models.seizure_transformer.detect()`. The model code itself is unchanged.

**Monte Carlo dropout:** SeizureTransformer registers `compute_agreement()` for optional uncertainty (`--enable-mcd`). Dropout is active during T stochastic forwards; BatchNorm stays in eval mode.

**Band-pass:** This package applies **0.5–120 Hz** Butterworth band-pass in `utils.py` (before notch). RobustSeiz can apply a **different band-pass on raw µV** (`--shift bandpass`) before inference: use this to test sensitivity to acquisition filter settings.

**Notch:** This package applies **1 Hz and 60 Hz** IIR notches after band-pass in `utils.py`. RobustSeiz can additionally apply notch on **raw µV** (`--shift notch`) before inference: use this to test whether built-in preprocessing masks power-line robustness issues.
