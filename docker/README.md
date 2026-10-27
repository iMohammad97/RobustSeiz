# Docker

GPU container for **RobustSeiz** inference: **ExperimentsUtils + active Model**.

## Build

```bash
docker compose -f docker/compose.yaml build
```

Builds the model named in `Model/active_model` (default: `SeizureTransformer` sample model).

Use a different model:

```bash
docker compose -f docker/compose.yaml build --build-arg ACTIVE_MODEL=YourModel
```

## Entry point

```bash
python3 -m experiments_utils /data/<edf_rel> /output/<tsv_rel>
```

Shifts come from `-e` env vars set by `run_experiment.py`. The model receives already-perturbed EEG data.

Optional MCD env vars: `ENABLE_MCD=1`, `MCD_SAMPLES`, `MCD_THRESH`, `MCD_SUBSET`, `MCD_SUBSET_MODE` (`random` or `research`), `MCD_SUBSET_SEED`. When enabled, inference also writes `*_agreement.npz` next to each `*_events.tsv`.

## Manual run

```bash
docker compose -f docker/compose.yaml run --rm \
  -v /path/to/Data/ConvertedTUSZEval:/data:ro \
  -v /path/to/Results/MyExp:/output:rw \
  -e INPUT=sub-001/ses-01/eeg/sub-001_ses-01_task-szMonitoring_run-01.edf \
  -e OUTPUT=sub-001/ses-01/eeg/sub-001_ses-01_task-szMonitoring_run-01_events.tsv \
  -e PERTURB=awgn -e SNR_DB=20 \
  main
```

Notch (mismatch: 60 Hz mains + 50 Hz notch):

```bash
docker compose -f docker/compose.yaml run --rm \
  -v /path/to/Data/ConvertedTUSZEval:/data:ro \
  -v /path/to/Results/NOTCH_mismatch:/output:rw \
  -e INPUT=sub-001/.../file.edf \
  -e OUTPUT=sub-001/.../file_events.tsv \
  -e PERTURB=notch -e NOTCH_MODE=mismatch -e NOTCH_PROFILE=50 \
  -e MAINS_HZ=60 -e AMP_RATIO=0.03 \
  main
```

## Requirements

- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Model weights in place before build (SeizureTransformer: see `Model/README.md`; EEGWaveNet: fetch it first, see `Model/EEGWaveNet/README.md`)

## What gets installed

1. `ExperimentsUtils`: shifts + inference + model registry
2. Every package under `Model/<active_model>/` (each folder with a `pyproject.toml`): detector only

The image has no data or results baked in: `run_experiment.py` mounts the chosen dataset at `/data` and the run's output folder at `/output` for every recording. Without a GPU, use `--runner local` instead.


Band-pass (0.5–40 Hz on raw EEG):

```bash
docker compose -f docker/compose.yaml run --rm \
  -v /path/to/Data/ConvertedTUSZEval:/data:ro \
  -v /path/to/Results/BP_0p5_40:/output:rw \
  -e INPUT=sub-001/.../file.edf \
  -e OUTPUT=sub-001/.../file_events.tsv \
  -e SHIFT=bandpass -e BANDPASS_PROFILE=0.5_40 \
  main
```
