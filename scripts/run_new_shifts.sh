#!/usr/bin/env bash
# Run all newly implemented (not yet scored) experiment variations.
# From repo root. Requires: Docker, python/py, Data/ConvertedTUSZEval, model weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${RUN:-python run_experiment.py}"
# On Windows Git Bash you may use: RUN="py run_experiment.py"

echo "=== Building Docker image (once) ==="
docker compose -f docker/compose.yaml build

run() {
  echo ""
  echo ">>> $*"
  $RUN "$@"
}

# ---------------------------------------------------------------------------
# Environment: symmetric quant + random channel drop
# ---------------------------------------------------------------------------
run --category environment --shift quant --bits 12
run --category environment --shift drop --n-drop 1 --rng-seed 42
run --category environment --shift drop --n-drop 2 --rng-seed 42
run --category environment --shift drop --n-drop 3 --rng-seed 42

# ---------------------------------------------------------------------------
# Environment: band-pass acquisition (4 profiles)
# ---------------------------------------------------------------------------
for prof in 0.5_40 1_40 1_30 0.5_25; do
  run --category environment --shift bandpass --bandpass-profile "$prof"
done

# ---------------------------------------------------------------------------
# Environment: impedance B1 — full grid (bands × µV amplitudes, all channels)
# ---------------------------------------------------------------------------
for band in 0_1 0.05_0.5 0.1_1; do
  for uv in 25 50 100 200; do
    run --category environment --shift impedance \
      --impedance-band "$band" --amp-mode uv --amp-uv "$uv" --channel-mode all --rng-seed 42
  done
done

# Impedance: ratio amplitude (all channels)
for band in 0_1 0.05_0.5 0.1_1; do
  run --category environment --shift impedance \
    --impedance-band "$band" --amp-mode ratio --amp-ratio 0.25 --channel-mode all --rng-seed 42
done

# Impedance: spatial patterns (representative subset)
run --category environment --shift impedance --impedance-band 0_1 --amp-mode uv --amp-uv 50 --channel-mode one --rng-seed 42
run --category environment --shift impedance --impedance-band 0_1 --amp-mode uv --amp-uv 50 --channel-mode subset --n-channels 3 --rng-seed 42
run --category environment --shift impedance --impedance-band 0.05_0.5 --amp-mode uv --amp-uv 100 --channel-mode subset --n-channels 2 --rng-seed 42

# ---------------------------------------------------------------------------
# Environment: baseline drift B2
# ---------------------------------------------------------------------------
# Legacy sin mode (single frequency)
run --category environment --shift drift --drift-mode sin --drift-hz 0.2 --amp-mode ratio --amp-ratio 0.03 --channel-mode all --rng-seed 42
run --category environment --shift drift --drift-mode sin --drift-hz 0.2 --amp-mode uv --amp-uv 50 --channel-mode all --rng-seed 42

# sin_band + random_walk per drift band
for band in 0_1 0.05_0.5 0.1_1; do
  for mode in sin_band random_walk; do
    run --category environment --shift drift \
      --drift-mode "$mode" --drift-band "$band" \
      --amp-mode ratio --amp-ratio 0.10 --channel-mode all --rng-seed 42
    run --category environment --shift drift \
      --drift-mode "$mode" --drift-band "$band" \
      --amp-mode uv --amp-uv 50 --channel-mode all --rng-seed 42
  done
done

# piecewise drift
run --category environment --shift drift --drift-mode piecewise --drift-band 0.05_0.5 \
  --amp-mode ratio --amp-ratio 0.10 --channel-mode all --piecewise-segments 8 --rng-seed 42

# intermittent drift
run --category environment --shift drift --drift-mode random_walk --drift-band 0.05_0.5 \
  --amp-mode ratio --amp-ratio 0.10 --channel-mode all --intermittent --intermittent-duty 0.3 --rng-seed 42
run --category environment --shift drift --drift-mode sin_band --drift-band 0.1_1 \
  --amp-mode uv --amp-uv 100 --channel-mode all --intermittent --intermittent-duty 0.3 --rng-seed 42

# drift spatial patterns
run --category environment --shift drift --drift-mode random_walk --drift-band 0.05_0.5 \
  --amp-mode ratio --amp-ratio 0.10 --channel-mode one --rng-seed 42
run --category environment --shift drift --drift-mode sin_band --drift-band 0_1 \
  --amp-mode uv --amp-uv 50 --channel-mode subset --n-channels 2 --rng-seed 42

# ---------------------------------------------------------------------------
# Noise: notch filter — only mode (all profiles)
# ---------------------------------------------------------------------------
for prof in off 50 60 50_100 60_120; do
  run --category noise --shift notch --notch-profile "$prof" --notch-mode only
done

# Notch mismatch: mains + wrong notch
run --category noise --shift notch --notch-mode mismatch --mains-hz 60 --notch-profile 50 --amp-ratio 0.03
run --category noise --shift notch --notch-mode mismatch --mains-hz 50 --notch-profile 60 --amp-ratio 0.03
run --category noise --shift notch --notch-mode mismatch --mains-hz 60 --notch-profile 50_100 --amp-ratio 0.05
run --category noise --shift notch --notch-mode mismatch --mains-hz 50 --notch-profile 60_120 --amp-ratio 0.05
run --category noise --shift notch --notch-mode mismatch --mains-hz 60 --notch-profile 50 --amp-ratio 0.10

echo ""
echo "=== All new-shift experiments finished ==="
