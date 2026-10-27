#!/usr/bin/env python3
"""Convert raw Siena / CHB-MIT PhysioNet releases to BIDS-EEG with epilepsy2bids.

Usage:
    python scripts/convert_datasets.py siena  <raw_root> Data/ConvertedSiena
    python scripts/convert_datasets.py chbmit <raw_root> Data/ConvertedCHBMIT

<raw_root> is the inner release folder (the one containing PN*/ or chb*/).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _patch_siena_timestamps() -> None:
    """Make the Siena annotation parser work on Windows.

    epilepsy2bids parses ``HH.MM.SS`` with ``time.mktime``, which raises OverflowError on
    Windows for the implied year 1900. Seconds since midnight give the same differences.
    """
    import epilepsy2bids.load_annotations.siena as siena

    def _parse_time_stamp(string: str) -> float:
        stamp = re.findall(".*[ |:]([0-9]+.[0-9]+.[0-9]+).*", string)[0]
        h, m, s = (int(v) for v in re.split(r"[.:]", stamp))
        return float(h * 3600 + m * 60 + s)

    siena._parseTimeStamp = _parse_time_stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["siena", "chbmit"])
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"{args.out_dir} is not empty; refusing to overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "siena":
        _patch_siena_timestamps()
        from epilepsy2bids.bids.siena.convert2bids import convert
    else:
        from epilepsy2bids.bids.chbmit.convert2bids import convert

    convert(args.raw_root.resolve(), args.out_dir.resolve())
    print(f"Converted {args.dataset} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
