"""Score front/back tracer outputs against the local multi-wrap reference.

The headline definition is unchanged from the accepted v4 audit: a point is
hop-correct when its nearest reference class is the expected adjacent class
and its distance to that class is below half the cell's local inter-wrap gap.
Reference-boundary cells are excluded. This publication version replaces the
original session-specific scratch paths with a validated, reproducible CLI.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_core import (
    VOX_UM,
    load_grid,
    load_reference,
    render_status,
    score_prediction,
    summarize_score,
    wrap_error_histogram,
    write_png,
)


def result_summary(result):
    summary = summarize_score(result)
    ok = result["ok"]
    summary.update(
        {
            "median_expected_dist_vox": float(np.median(result["d_exp"][ok])),
            "p90_expected_dist_vox": float(np.percentile(result["d_exp"][ok], 90)),
            "median_local_gap_vox": float(np.median(result["gap"][ok])),
            "p10_local_gap_vox": float(np.percentile(result["gap"][ok], 10)),
            "p90_local_gap_vox": float(np.percentile(result["gap"][ok], 90)),
            "median_expected_dist_gap_ratio": float(np.median(result["ratio"][ok])),
            "wrap_index_error_histogram": wrap_error_histogram(result),
        }
    )
    return summary


def print_result(label, result):
    summary = result_summary(result)
    included = summary["included"]
    print(
        f"[{label}] scorable={included} excluded={summary['excluded']} "
        f"hop_correct={summary['correct']}/{included} "
        f"({summary['correct_pct']:.3f}%) wrong_hop={summary['wrong_hop']}/{included} "
        f"({summary['wrong_hop_pct']:.3f}%) wrong_wrap={summary['wrong_wrap']} "
        f"distance_miss={summary['distance_miss']}",
        flush=True,
    )
    print(
        f"  expected distance median={summary['median_expected_dist_vox']:.3f} vox "
        f"({summary['median_expected_dist_vox'] * VOX_UM:.1f} um) "
        f"p90={summary['p90_expected_dist_vox']:.3f}; "
        f"local gap median={summary['median_local_gap_vox']:.3f} vox "
        f"p10={summary['p10_local_gap_vox']:.3f} "
        f"p90={summary['p90_local_gap_vox']:.3f}; "
        f"median distance/gap={summary['median_expected_dist_gap_ratio']:.3f}",
        flush=True,
    )
    print(
        f"  signed nearest-class error histogram: "
        f"{summary['wrap_index_error_histogram']}",
        flush=True,
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="band_r1145_200_xyz.npz")
    parser.add_argument("--normals", default="seed_normals.npy")
    parser.add_argument("--front", default="out_mps/seed_segment_front")
    parser.add_argument("--back", default="out_mps/seed_segment_back")
    parser.add_argument("--label", default="reduced")
    parser.add_argument("--out", default="figures/winding_failure_map_v4.png")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--no-controls", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ref = load_reference(args.reference)
    front = load_grid(args.front)
    back = load_grid(args.back)

    seed_points = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    self_dist, _ = ref.trees[0].query(seed_points, workers=-1)
    if float(self_dist.max()) != 0.0 or not np.all(ref.seed_cls == 0):
        raise AssertionError("reference self-test failed")
    print(
        f"SELF_TEST seed_class0={int((ref.seed_cls == 0).sum())}/{ref.seed_cls.size} "
        f"median={np.median(self_dist):.3f} max={self_dist.max():.3f} vox",
        flush=True,
    )

    front_result = score_prediction(ref, front, +1)
    back_result = score_prediction(ref, back, -1)
    summaries = {
        "front": print_result(f"{args.label} front", front_result),
        "back": print_result(f"{args.label} back", back_result),
    }

    if not args.no_controls:
        normals = np.asarray(np.load(args.normals), dtype=np.float64)
        if normals.shape != ref.seed.shape:
            raise ValueError("normal grid must match the bundled seed shape")
        norms = np.linalg.norm(normals, axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("normal grid must contain unit vectors")
        front_magnitude = float(np.median(np.linalg.norm(front - ref.seed, axis=-1)))
        back_magnitude = float(np.median(np.linalg.norm(back - ref.seed, axis=-1)))
        controls = {
            "constant_offset_front": score_prediction(
                ref, ref.seed - front_magnitude * normals, +1
            ),
            "constant_offset_back": score_prediction(
                ref, ref.seed + back_magnitude * normals, -1
            ),
            "front_scored_as_back": score_prediction(ref, front, -1),
        }
        for name, result in controls.items():
            summaries[name] = print_result(name, result)

    shape = ref.rr.shape
    gap = np.full((shape[0] * 4, 8, 3), 255, dtype=np.uint8)
    image = np.concatenate(
        [render_status(front_result["status"], shape), gap,
         render_status(back_result["status"], shape)],
        axis=1,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not write_png(args.out, image):
        raise OSError(f"failed to write {args.out}")
    print(f"wrote {args.out}", flush=True)

    if args.summary_json:
        payload = {
            "config": args.label,
            "reference_self_test": {
                "count": int(self_dist.size),
                "median_vox": float(np.median(self_dist)),
                "max_vox": float(self_dist.max()),
            },
            "results": summaries,
        }
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        print(f"wrote {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
