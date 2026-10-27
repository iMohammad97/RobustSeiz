# EEGWaveNet: Multiscale CNN-Based Spatiotemporal Feature Extraction for EEG Seizure Detection

SzCORE compatible reproduction of [Thuwajit et al., 'EEGWaveNet: Multiscale CNN-Based Spatiotemporal Feature Extraction for EEG Seizure Detection', 2021](https://doi.org/10.1109/TII.2021.3133307), by EPFL ([esl-epfl/eegwavenet_2021](https://github.com/esl-epfl/eegwavenet_2021)).

The upstream repository has no licence terms (its `LICENSE` file is empty), so its code and weights are not redistributed here. Fetch the package at the pinned commit (`UPSTREAM_COMMIT`) instead:

```bash
git clone https://github.com/esl-epfl/eegwavenet_2021 /tmp/eegwavenet_2021
git -C /tmp/eegwavenet_2021 checkout 7e630abb542fb43366960da6e475ae705f1d6450
cp -r /tmp/eegwavenet_2021/thuwajit_2021 Model/EEGWaveNet/thuwajit_2021
pip install -e Model/EEGWaveNet/thuwajit_2021
```

`Model/EEGWaveNet/thuwajit_2021/` is gitignored. Model details and dataset coverage: `MODEL.md`.
