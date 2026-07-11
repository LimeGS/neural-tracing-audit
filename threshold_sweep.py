"""Threshold sweeps for the local multi-wrap reference benchmark.

This computes hop-correct / wrong-hop rates under the same exclusion and
nearest-surface assignment rules as the headline half-gap metric, but lets
the distance threshold vary by gap fraction, fixed voxels, or fixed microns.
"""

import argparse
import csv
import sys

from benchmark_core import (
    VOX_UM,
    load_grid,
    load_reference,
    score_prediction,
    summarize_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default="band_r1145_200_xyz.npz")
    parser.add_argument("--front", default="gate_3090/out_native/seed_segment_front")
    parser.add_argument("--back", default="gate_3090/out_native/seed_segment_back")
    parser.add_argument("--label", default="native")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--gap-fractions",
        nargs="*",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.5],
    )
    parser.add_argument(
        "--fixed-vox",
        nargs="*",
        type=float,
        default=[2, 4, 6, 8, 10, 12, 16, 24, 32],
    )
    parser.add_argument(
        "--fixed-um",
        nargs="*",
        type=float,
        default=[25, 50, 75, 100, 150, 200],
    )
    return parser.parse_args()


def row_for(label, direction, kind, display_value, result):
    summary = summarize_score(result)
    included = summary["included"]
    median_dist = ""
    p90_dist = ""
    median_gap = ""
    median_threshold = ""
    if included:
        ok = result["ok"]
        import numpy as np

        median_dist = f"{float(np.median(result['d_exp'][ok])):.3f}"
        p90_dist = f"{float(np.percentile(result['d_exp'][ok], 90)):.3f}"
        median_gap = f"{float(np.median(result['gap'][ok])):.3f}"
        median_threshold = f"{float(np.median(result['threshold'][ok])):.3f}"
    return {
        "config": label,
        "direction": direction,
        "threshold_kind": kind,
        "threshold_value": display_value,
        "median_threshold_vox": median_threshold,
        "scorable": included,
        "excluded": summary["excluded"],
        "hop_correct": summary["correct"],
        "wrong_hop": summary["wrong_hop"],
        "wrong_adjacent_wrap_identity": summary["wrong_wrap"],
        "correct_wrap_distance_miss": summary["distance_miss"],
        "hop_correct_pct": f"{summary['correct_pct']:.3f}",
        "wrong_hop_pct": f"{summary['wrong_hop_pct']:.3f}",
        "wrong_wrap_pct": f"{summary['wrong_wrap_pct']:.3f}",
        "distance_miss_pct": f"{summary['distance_miss_pct']:.3f}",
        "median_expected_dist_vox": median_dist,
        "p90_expected_dist_vox": p90_dist,
        "median_local_gap_vox": median_gap,
    }


def main():
    args = parse_args()
    ref = load_reference(args.npz)
    grids = {
        "front": (load_grid(args.front), +1),
        "back": (load_grid(args.back), -1),
    }

    rows = []
    for direction, (grid, expected_class) in grids.items():
        for frac in args.gap_fractions:
            result = score_prediction(ref, grid, expected_class, "gap_fraction", frac)
            rows.append(row_for(args.label, direction, "gap_fraction", frac, result))
        for vox in args.fixed_vox:
            result = score_prediction(ref, grid, expected_class, "fixed_vox", vox)
            rows.append(row_for(args.label, direction, "fixed_vox", vox, result))
        for um in args.fixed_um:
            vox = um / VOX_UM
            result = score_prediction(ref, grid, expected_class, "fixed_vox", vox)
            rows.append(row_for(args.label, direction, "fixed_um", um, result))

    fieldnames = list(rows[0].keys())
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
