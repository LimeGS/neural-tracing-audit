"""Validate bundled tracer outputs and report displacement statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_core import CX, CY, VOX_UM, gaps_for, load_grid, load_reference


def valid_grid(grid):
    return np.isfinite(grid).all(axis=-1) & ~np.all(grid == -1, axis=-1)


def displacement_summary(seed, output):
    if output.shape != seed.shape:
        raise ValueError(f"output shape {output.shape} does not match seed {seed.shape}")
    valid = valid_grid(seed) & valid_grid(output)
    if not valid.any():
        raise ValueError("seed and output have no mutually valid cells")
    delta = output - seed
    distance = np.linalg.norm(delta, axis=-1)
    r_seed = np.hypot(seed[..., 0] - CX, seed[..., 1] - CY)
    r_output = np.hypot(output[..., 0] - CX, output[..., 1] - CY)
    radial_shift = r_output - r_seed
    return {
        "valid": int(valid.sum()),
        "total": int(valid.size),
        "non_finite_values": int((~np.isfinite(output)).sum()),
        "distance_mean_vox": float(distance[valid].mean()),
        "distance_median_vox": float(np.median(distance[valid])),
        "distance_p5_vox": float(np.percentile(distance[valid], 5)),
        "distance_p95_vox": float(np.percentile(distance[valid], 95)),
        "distance_max_vox": float(distance[valid].max()),
        "radial_shift_mean_vox": float(radial_shift[valid].mean()),
        "radial_shift_abs_mean_vox": float(np.abs(radial_shift[valid]).mean()),
        "valid_mask": valid,
        "distance_grid": distance,
    }


def write_heatmap(path, summary):
    valid = summary["valid_mask"]
    distance = summary["distance_grid"]
    vmax = max(float(distance[valid].max()), 1e-6)
    normalized = np.clip(distance / vmax * 255.0, 0, 255).astype(np.uint8)
    value = normalized.astype(np.float32) / 255.0
    color = np.stack(
        [255 * value, 255 * np.sqrt(value), 255 * (1.0 - value)], axis=-1
    ).astype(np.uint8)
    color[~valid] = (32, 32, 32)
    image = Image.fromarray(color).resize(
        (color.shape[1] * 3, color.shape[0] * 3), Image.Resampling.NEAREST
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="band_r1145_200_xyz.npz")
    parser.add_argument("--seed", default="seed_segment")
    parser.add_argument("--front", default="out_mps/seed_segment_front")
    parser.add_argument("--back", default="out_mps/seed_segment_back")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed = load_grid(args.seed)
    ref = load_reference(args.reference)
    if not np.array_equal(seed.astype(np.float32), ref.seed.astype(np.float32)):
        raise ValueError("seed tifxyz does not exactly match the reference-band crop")

    payload = {}
    for direction, path in (("front", args.front), ("back", args.back)):
        summary = displacement_summary(seed, load_grid(path))
        print(
            f"{direction}: valid={summary['valid']}/{summary['total']} "
            f"non_finite={summary['non_finite_values']} "
            f"distance median={summary['distance_median_vox']:.3f} vox "
            f"({summary['distance_median_vox'] * VOX_UM:.1f} um) "
            f"p5={summary['distance_p5_vox']:.3f} "
            f"p95={summary['distance_p95_vox']:.3f} "
            f"max={summary['distance_max_vox']:.3f} "
            f"radial_mean={summary['radial_shift_mean_vox']:.3f}",
            flush=True,
        )
        write_heatmap(Path(args.out_dir) / f"displacement_{direction}.png", summary)
        payload[direction] = {
            key: value for key, value in summary.items()
            if key not in {"valid_mask", "distance_grid"}
        }

    for direction, expected in (("front", +1), ("back", -1)):
        gap, _, edge = gaps_for(ref, expected)
        usable = ~edge & (ref.seed_cls == 0)
        print(
            f"{direction} local-reference gap median={np.median(gap[usable]):.3f} vox "
            f"({np.median(gap[usable]) * VOX_UM:.1f} um)",
            flush=True,
        )

    if args.summary_json:
        with open(args.summary_json, "w") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        print(f"wrote {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
