"""Compatibility entry point for scoring the bundled native-config outputs."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from winding_audit_v4 import main as score_main  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=str(ROOT / "band_r1145_200_xyz.npz"))
    parser.add_argument(
        "--front", default=str(ROOT / "gate_3090/out_native/seed_segment_front")
    )
    parser.add_argument(
        "--back", default=str(ROOT / "gate_3090/out_native/seed_segment_back")
    )
    parser.add_argument(
        "--out", default=str(ROOT / "figures/winding_failure_map_native.png")
    )
    parser.add_argument("--summary-json", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    forwarded = [
        "--reference", args.reference,
        "--front", args.front,
        "--back", args.back,
        "--label", "native",
        "--out", args.out,
        "--no-controls",
    ]
    if args.summary_json:
        forwarded.extend(["--summary-json", args.summary_json])
    score_main(forwarded)


if __name__ == "__main__":
    main()
