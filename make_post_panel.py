"""Render the community-post illustration panel (front cell r=108, c=134).

One CT cutaway with four markers telling the resolution-calibration story:

- cyan:    seed point on the current wrap
- yellow:  recorded surface of the adjacent wrap (the target)
- magenta: prediction as released (unit-mismatched, overshoots into the gap)
- green:   the SAME prediction with displacement x 4.8/7.91 (lands on target)

The cut plane contains both the seed and the prediction (no off-plane
projection for either; the reference point is ~1-2 vox off-plane). Unlike
figures/ct_panels/ (whose cells are selected by fixed scoring rules), this
cell was hand-picked for CT legibility as an ILLUSTRATION: a typical
pure-unit miss (as-released distance miss 0.79x the local gap; calibrated
0.09x), in a spot where the sheets render cleanly. All quantitative claims
come from the scorer over the full window, not from this panel.

Run from the repository root:  python make_post_panel.py
Requires band_r1145_200_xyz.npz (see README) and network access to the
public Scroll 3 zarr.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_core import load_reference, load_grid, score_prediction
from resolution_calibration import rescale_prediction, FACTOR
from make_ct_panels import local_axes, open_volume, sample_ct_plane, project, DEFAULT_ZARR

R, C = 108, 134
SIZE, WIDTH_VOX, HEIGHT_VOX = 480, 88, 120
OUT = "figures/post_panel_front_r108_c134.png"
TEXT = (255, 214, 10)  # subtitle yellow, drawn with a black outline


def load_font(size):
    try:  # DejaVu ships with matplotlib, already a dependency
        import matplotlib
        return ImageFont.truetype(
            str(matplotlib.get_data_path()) + "/fonts/ttf/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def main():
    ref = load_reference("band_r1145_200_xyz.npz")
    grid = load_grid("gate_3090/out_native/seed_segment_front")
    res = score_prediction(ref, grid, +1)
    cal = score_prediction(ref, rescale_prediction(ref.seed, grid, FACTOR), +1)
    flat = np.where((ref.rr.ravel() == R) & (ref.cc.ravel() == C))[0][0]

    seed = ref.seed[R, C]
    pred = res["pred_points"][flat]
    calp = cal["pred_points"][flat]
    rpt = ref.pts_of[+1][cal["idx_exp"][flat]]

    _, av = local_axes(ref.seed, R, C, pred, seed)
    rel = pred - seed
    u_vec = rel - np.dot(rel, av) * av
    au = u_vec / np.linalg.norm(u_vec)

    vol = open_volume(DEFAULT_ZARR, "0")
    panel, ext = sample_ct_plane(vol, seed, au, av, WIDTH_VOX, HEIGHT_VOX, SIZE)
    img = Image.fromarray(np.repeat(panel[..., None], 3, axis=2))
    d = ImageDraw.Draw(img)

    def mark(point, color, rad):
        x, y = project(point, seed, au, av, ext, SIZE)
        d.ellipse((x - rad, y - rad, x + rad, y + rad), outline=color, width=3)
        d.line((x - rad, y, x + rad, y), fill=color, width=2)
        d.line((x, y - rad, x, y + rad), fill=color, width=2)

    mark(seed, (0, 255, 255), 10)
    mark(rpt, (255, 230, 0), 10)
    mark(pred, (255, 0, 255), 10)
    mark(calp, (0, 255, 60), 19)

    font = load_font(16)
    small = load_font(14)

    def caption(xy, text, f=font):
        d.text(xy, text, fill=TEXT, font=f, stroke_width=2, stroke_fill=(0, 0, 0))

    caption((10, 8), f"front distance miss, cell r={R} c={C}")
    caption((10, 32), "cyan seed  |  yellow next-wrap surface")
    caption((10, 56), "magenta pred as released")
    caption((10, 80), "green same pred x 4.8/7.91")

    bar = int(round(10 * SIZE / WIDTH_VOX))
    label = "10 vox = 79 um"
    d.line((SIZE - 20 - bar, SIZE - 18, SIZE - 20, SIZE - 18), fill=TEXT, width=4)
    caption((SIZE - 20 - d.textlength(label, font=small), SIZE - 42), label, small)

    img.save(OUT)
    print(
        f"wrote {OUT}\n"
        f"local gap {res['gap'][flat]:.1f} vox | as released: miss {res['d_exp'][flat]:.1f} vox"
        f" ({res['ratio'][flat]:.2f}x gap) | calibrated: {cal['d_exp'][flat]:.1f} vox"
        f" ({cal['ratio'][flat]:.2f}x gap), correct={bool(cal['correct'][flat])}"
    )


if __name__ == "__main__":
    main()
