"""Generate CT cutaway panels for representative benchmark cells.

The script selects pass, borderline half-gap miss, severe miss, wrong-wrap
identity miss (if present), and excluded-by-edge cells from the same local
multi-wrap scoring logic used by the benchmark. It then renders oblique CT
planes through each seed cell and overlays:

- cyan: seed point
- magenta: predicted point
- yellow: nearest point on the expected local-reference wrap

The default CT source is the public Scroll 3 53 keV zarr and therefore
requires network access unless you pass a local zarr path.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from benchmark_core import (
    C0,
    VOX_UM,
    load_grid,
    load_reference,
    score_prediction,
)


DEFAULT_ZARR = (
    "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/"
    "volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default="band_r1145_200_xyz.npz")
    parser.add_argument("--front", default="gate_3090/out_native/seed_segment_front")
    parser.add_argument("--back", default="gate_3090/out_native/seed_segment_back")
    parser.add_argument("--volume", default=DEFAULT_ZARR)
    parser.add_argument("--level", default="0")
    parser.add_argument("--out-dir", default="figures/ct_panels")
    parser.add_argument("--max-panels", type=int, default=10)
    parser.add_argument("--plane-width", type=int, default=96)
    parser.add_argument("--plane-height", type=int, default=160)
    parser.add_argument("--size", type=int, default=240)
    return parser.parse_args()


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def local_axes(seed_grid, r, c, pred_point, seed_point):
    r0, r1 = max(r - 1, 0), min(r + 1, seed_grid.shape[0] - 1)
    c0, c1 = max(c - 1, 0), min(c + 1, seed_grid.shape[1] - 1)
    tangent_c = unit(seed_grid[r, c1] - seed_grid[r, c0])
    tangent_r = unit(seed_grid[r1, c] - seed_grid[r0, c])
    normal = unit(np.cross(tangent_c, tangent_r))
    if np.linalg.norm(normal) < 1e-9:
        normal = unit(pred_point - seed_point)
    if np.dot(pred_point - seed_point, normal) < 0:
        normal = -normal
    return tangent_c, normal


def open_volume(path, level):
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "make_ct_panels.py requires zarr. Install the benchmark dependencies "
            "with `pip install -r requirements.txt`, or run this in the same "
            "environment used for the tracer."
        ) from exc

    root = zarr.open(path, mode="r")
    try:
        return root[level]
    except Exception:
        return root


def sample_ct_plane(volume, center_xyz, axis_u, axis_v, width_vox, height_vox, size):
    u = np.linspace(-width_vox / 2, width_vox / 2, size)
    v = np.linspace(-height_vox / 2, height_vox / 2, size)
    uu, vv = np.meshgrid(u, v)
    pts = center_xyz[None, None, :] + uu[..., None] * axis_u + vv[..., None] * axis_v

    zi = pts[..., 2]
    yi = pts[..., 1]
    xi = pts[..., 0]
    z0 = max(int(np.floor(zi.min())) - 2, 0)
    y0 = max(int(np.floor(yi.min())) - 2, 0)
    x0 = max(int(np.floor(xi.min())) - 2, 0)
    z1 = min(int(np.ceil(zi.max())) + 3, volume.shape[0])
    y1 = min(int(np.ceil(yi.max())) + 3, volume.shape[1])
    x1 = min(int(np.ceil(xi.max())) + 3, volume.shape[2])

    block = np.asarray(volume[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    coords = np.vstack([
        (zi - z0).ravel(),
        (yi - y0).ravel(),
        (xi - x0).ravel(),
    ])
    panel = map_coordinates(block, coords, order=1, mode="nearest").reshape(size, size)
    lo, hi = np.percentile(panel, [1, 99])
    scaled = np.clip((panel - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8), (u[0], u[-1], v[0], v[-1])


def project(point, center, axis_u, axis_v, extents, size):
    u0, u1, v0, v1 = extents
    rel = point - center
    pu = float(np.dot(rel, axis_u))
    pv = float(np.dot(rel, axis_v))
    x = int(round((pu - u0) / (u1 - u0) * (size - 1)))
    y = int(round((v1 - pv) / (v1 - v0) * (size - 1)))
    return x, y


def draw_marker(draw, xy, color, radius=5):
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    draw.line((x - radius, y, x + radius, y), fill=color, width=1)
    draw.line((x, y - radius, x, y + radius), fill=color, width=1)


def choose(mask, values, target=None, mode="closest"):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    if mode == "max":
        return int(idx[np.argmax(values[idx])])
    if target is None:
        return int(idx[0])
    return int(idx[np.argmin(np.abs(values[idx] - target))])


def representatives(ref, result, direction):
    ratio = result["ratio"]
    rr = ref.rr.ravel()
    cc = ref.cc.ravel()
    center_dist = (rr - 100) ** 2 + (cc - 100) ** 2
    excluded_choice = choose(result["excluded"], -center_dist, mode="max")
    reps = [
        ("pass", choose(result["correct"], ratio, target=0.25), direction),
        ("borderline_half_gap_miss", choose(result["distance_miss"], ratio, target=0.55), direction),
        ("severe_distance_miss", choose(result["distance_miss"], ratio, mode="max"), direction),
        ("wrong_wrap_identity", choose(result["wrong_wrap"], ratio, mode="max"), direction),
        ("excluded_by_edge", excluded_choice, direction),
    ]
    return [r for r in reps if r[1] is not None]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = load_reference(args.npz)
    grids = {
        "front": (load_grid(args.front), +1),
        "back": (load_grid(args.back), -1),
    }
    scored = {
        direction: (grid, expected, score_prediction(ref, grid, expected))
        for direction, (grid, expected) in grids.items()
    }
    volume = open_volume(args.volume, args.level)

    selected = []
    for direction, (_, _, result) in scored.items():
        selected.extend(representatives(ref, result, direction))
    selected = selected[: args.max_panels]

    manifest = []
    seed_grid = ref.seed
    rr = ref.rr.ravel()
    cc = ref.cc.ravel()
    for category, flat_idx, direction in selected:
        grid, expected, result = scored[direction]
        r = int(rr[flat_idx])
        c = int(cc[flat_idx])
        seed_point = seed_grid[r, c]
        pred_point = result["pred_points"][flat_idx]
        ref_point = ref.pts_of[expected][result["idx_exp"][flat_idx]]
        axis_u, axis_v = local_axes(seed_grid, r, c, pred_point, seed_point)

        panel, extents = sample_ct_plane(
            volume,
            seed_point,
            axis_u,
            axis_v,
            args.plane_width,
            args.plane_height,
            args.size,
        )
        rgb = np.repeat(panel[..., None], 3, axis=2)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        draw_marker(draw, project(seed_point, seed_point, axis_u, axis_v, extents, args.size), (0, 255, 255))
        draw_marker(draw, project(pred_point, seed_point, axis_u, axis_v, extents, args.size), (255, 0, 255))
        draw_marker(draw, project(ref_point, seed_point, axis_u, axis_v, extents, args.size), (255, 230, 0))
        draw.text((6, 6), f"{direction} {category} r={r} c={c}", fill=(255, 255, 255))
        draw.text((6, 22), "cyan seed | magenta pred | yellow ref", fill=(255, 255, 255))

        filename = f"{direction}_{category}_r{r:03d}_c{c:03d}.png"
        img.save(out_dir / filename)
        manifest.append({
            "file": filename,
            "direction": direction,
            "category": category,
            "seed_row": r,
            "seed_col": c,
            "global_col": c + C0,
            "status_code": int(result["status"][flat_idx]),
            "distance_vox": f"{float(result['d_exp'][flat_idx]):.3f}",
            "distance_um": f"{float(result['d_exp'][flat_idx] * VOX_UM):.1f}",
            "local_gap_vox": f"{float(result['gap'][flat_idx]):.3f}",
            "ratio_to_gap": f"{float(result['ratio'][flat_idx]):.3f}",
            "nearest_class": int(result["nearest_cls"][flat_idx]),
            "expected_class": expected,
        })

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(manifest[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest)

    print(f"wrote {len(manifest)} panels to {out_dir}")


if __name__ == "__main__":
    main()
