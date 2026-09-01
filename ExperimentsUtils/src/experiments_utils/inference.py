from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg

from experiments_utils.apply import apply_shifts_from_env
from experiments_utils.mcd import agreement_output_path
from experiments_utils.models.registry import get_active_model_name, get_mcd_fn, get_model_fn, model_supports_mcd


def _mcd_enabled() -> bool:
    return os.getenv("ENABLE_MCD", "").lower() in ("1", "true", "yes")


def run_inference(edf_file: str, out_file: str, project_root: Path | None = None) -> None:
    """Benchmark entry point: apply shifts, run active model, write BIDS TSV."""
    eeg = Eeg.loadEdf(edfFile=edf_file)
    assert eeg.montage is Eeg.Montage.UNIPOLAR, "Error: Only unipolar montages are supported."

    labels = getattr(eeg, "channels", None)
    print("eeg.channels:", labels)

    data = apply_shifts_from_env(eeg.data, eeg.fs, labels, raw_data=eeg.data)

    detect = get_model_fn(project_root)
    mask = detect(eeg, data)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hyp = Annotations.loadMask(mask, eeg.fs)
    hyp.saveTsv(str(out_path))

    if _mcd_enabled():
        model_name = get_active_model_name(project_root)
        if not model_supports_mcd(project_root):
            print(
                f"WARNING: ENABLE_MCD is set but model {model_name!r} does not "
                "register Monte Carlo dropout support; skipping uncertainty output."
            )
            return

        mcd_fn = get_mcd_fn(project_root)
        assert mcd_fn is not None
        agreement = mcd_fn(eeg, data)
        agreement_path = agreement_output_path(out_path)
        np.savez(agreement_path, agreement=agreement, fs=float(eeg.fs))
        print(
            f"[mcd] saved agreement trace to {agreement_path} "
            f"(mean={float(np.mean(agreement)):.4f}, T={os.getenv('MCD_SAMPLES', '20')})"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Robustness benchmark inference: shifts + active model -> BIDS TSV."
    )
    parser.add_argument("input", help="Path to input EDF.")
    parser.add_argument("output", help="Path to output events TSV.")
    args = parser.parse_args()
    run_inference(args.input, args.output)


if __name__ == "__main__":
    main()
