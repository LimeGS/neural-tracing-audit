"""Cross-check local reference gaps against CT intensity peak spacing.

This command uses only bundled geometry/normals plus the public CT zarr. It
also reports signed normal and total displacement magnitudes for the reduced
tracer outputs. Peak spacing is an independent image-intensity check, not a
manual ground-truth annotation.
"""

import argparse
import csv

import numpy as np
from scipy.signal import find_peaks
import zarr

from benchmark_core import VOX_UM, load_grid, load_reference


DEFAULT_VOLUME = (
    "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/"
    "volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr"
)
DEFAULT_CELLS = [(r, c) for r in (30, 100, 170) for c in (30, 100, 170)]


def open_volume(path, level):
    root = zarr.open(path, mode="r")
    try:
        return root[level]
    except (KeyError, IndexError, TypeError):
        return root


def displacement_stats(seed, prediction, direction):
    displacement = prediction - seed
    signed = np.sum(displacement * direction, axis=-1)
    total = np.linalg.norm(displacement, axis=-1)
    tangential = np.sqrt(np.maximum(total**2 - signed**2, 0))
    return signed, tangential, total


def ct_profile(volume, seed_point, normal, offsets):
    points = seed_point[None, :] + offsets[:, None] * normal[None, :]
    zi = np.clip(np.round(points[:, 2]).astype(int), 0, volume.shape[0] - 1)
    yi = np.clip(np.round(points[:, 1]).astype(int), 0, volume.shape[1] - 1)
    xi = np.clip(np.round(points[:, 0]).astype(int), 0, volume.shape[2] - 1)
    z0, z1 = int(zi.min()), int(zi.max()) + 1
    y0, y1 = int(yi.min()), int(yi.max()) + 1
    x0, x1 = int(xi.min()), int(xi.max()) + 1
    block = np.asarray(volume[z0:z1, y0:y1, x0:x1])
    return block[zi - z0, yi - y0, xi - x0].astype(np.float64)


def peak_locations(profile, offsets, prominence=15.0, distance=5):
    smoothed = np.convolve(profile, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smoothed, prominence=prominence, distance=distance)
    return offsets[peaks]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="band_r1145_200_xyz.npz")
    parser.add_argument("--normals", default="seed_normals.npy")
    parser.add_argument("--front", default="out_mps/seed_segment_front")
    parser.add_argument("--back", default="out_mps/seed_segment_back")
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--level", default="0")
    parser.add_argument("--out", default="")
    parser.add_argument("--prominence", type=float, default=15.0)
    parser.add_argument("--peak-distance", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ref = load_reference(args.reference)
    normals = np.asarray(np.load(args.normals), dtype=np.float64)
    if normals.shape != ref.seed.shape:
        raise ValueError(f"normal shape {normals.shape} does not match seed {ref.seed.shape}")
    norms = np.linalg.norm(normals, axis=-1)
    if not np.isfinite(normals).all() or not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("seed normals must be finite unit vectors")

    predictions = {
        "front": (load_grid(args.front), -normals),
        "back": (load_grid(args.back), normals),
    }
    for name, (prediction, direction) in predictions.items():
        signed, tangential, total = displacement_stats(ref.seed, prediction, direction)
        print(
            f"NT {name}: signed normal offset median={np.median(signed):.3f} vox "
            f"({np.median(signed) * VOX_UM:.1f} um), "
            f"p10={np.percentile(signed, 10):.3f}, "
            f"p90={np.percentile(signed, 90):.3f}; "
            f"tangential median={np.median(tangential):.3f} vox; "
            f"total median={np.median(total):.3f} vox "
            f"({np.median(total) * VOX_UM:.1f} um)",
            flush=True,
        )

    volume = open_volume(args.volume, args.level)
    offsets = np.arange(-90.0, 90.5, 1.0)
    all_spacings = []
    records = []
    print("CT intensity peak spacing along bundled seed normals:", flush=True)
    for row, col in DEFAULT_CELLS:
        profile = ct_profile(volume, ref.seed[row, col], normals[row, col], offsets)
        peaks = peak_locations(
            profile, offsets, prominence=args.prominence, distance=args.peak_distance
        )
        spacings = np.diff(np.sort(peaks))
        all_spacings.extend(spacings.tolist())
        print(
            f"  cell ({row:3d},{col:3d}): peaks={peaks.tolist()} "
            f"spacings={spacings.tolist()}",
            flush=True,
        )
        records.append(
            {
                "seed_row": row,
                "seed_col": col,
                "peak_offsets_vox": ";".join(f"{value:.1f}" for value in peaks),
                "spacings_vox": ";".join(f"{value:.1f}" for value in spacings),
                "spacing_count": len(spacings),
            }
        )

    spacings = np.asarray(all_spacings, dtype=np.float64)
    if spacings.size == 0:
        raise RuntimeError("no CT intensity peak spacings were detected")
    print(
        f"CT_SPACING_DONE cells={len(DEFAULT_CELLS)} spacings={spacings.size} "
        f"median={np.median(spacings):.3f} vox "
        f"({np.median(spacings) * VOX_UM:.1f} um) "
        f"p10={np.percentile(spacings, 10):.3f} "
        f"p90={np.percentile(spacings, 90):.3f}",
        flush=True,
    )

    if args.out:
        with open(args.out, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
