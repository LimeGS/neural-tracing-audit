"""Resolution-calibration finding (2026-07-11): evidence generator.

The released checkpoint is reported as trained on 4.8 um voxels (see
docs/UPSTREAM_UNIT_AUDIT.md for the provenance boundary); the public PHerc0332
volume used for this benchmark is 7.91 um. The released CLI
(vesuvius==0.2.4, infer_rowcol_triplet_wraps) applies the model's
displacement output directly in input-volume voxels: no unit conversion
exists anywhere in vesuvius/neural_tracing (verified on the wheel:
displacements are added raw, with only a max-displacement clamp;
--volume-scale selects the zarr pyramid level; --tifxyz-voxel-size-um is
output metadata only), and no resolution requirement is documented.

Consequence: every predicted hop is physically inflated by 7.91/4.8 = 1.65x.
This script shows, with the public scorer:

  1. CONTROL: unscaled predictions reproduce the published numbers exactly.
  2. Scaling displacements by 4.8/7.91 (a physics-derived constant, not a
     fitted parameter) flips the benchmark: wrong-hop 63.2/56.3% -> ~5.9/7.7%,
     passing the pre-registered <30% gate with margin.
  3. The factor sits on a plateau (0.50-0.65 all pass), so it is not a
     hand-picked point.
  4. Adversarial controls on the calibrated predictions:
     - wrong-side scoring still fails completely (direction discrimination
       is intact);
     - BUT a saturation caveat: directions permuted within this low-curvature
       window and a reference-pitch normal-step diagnostic both score >90%.
       This window therefore certifies the magnitude fix, NOT learned
       directional skill over a coherent local normal; harder terrain
       (multi-hop chaining and curvature) is needed for that.

Run from the repository root:  python resolution_calibration.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_core import load_reference, load_grid, score_prediction, summarize_score

REF = os.path.join(HERE, "band_r1145_200_xyz.npz")
NATIVE = os.path.join(HERE, "gate_3090", "out_native")
NORMALS = os.path.join(HERE, "seed_normals.npy")
FACTOR = 4.8 / 7.91          # training voxel pitch / volume voxel pitch
PITCH_BINS = ((0.0, 7.0), (7.0, 9.0), (9.0, 12.0), (12.0, 16.0), (16.0, np.inf))


def pct(s):
    return f"correct {s['correct_pct']:.2f}% wrong-hop {s['wrong_hop_pct']:.2f}% (incl {s['included']})"


def rescale_prediction(seed, prediction, factor):
    """Rescale only the displacement vector, preserving seed coordinates."""
    seed = np.asarray(seed, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if seed.shape != prediction.shape:
        raise ValueError("seed and prediction must have identical shapes")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be finite and positive")
    return seed + (prediction - seed) * float(factor)


def summarize_on_mask(result, mask):
    """Summarize a score on a frozen denominator supplied by another run."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != result["ok"].shape:
        raise ValueError("mask shape does not match score result")
    newly_excluded = mask & result["excluded"]
    if newly_excluded.any():
        raise ValueError(
            f"rescore excludes {int(newly_excluded.sum())} points from the frozen denominator"
        )
    included = int(mask.sum())
    correct = int((result["correct"] & mask).sum())
    wrong_wrap = int((result["wrong_wrap"] & mask).sum())
    distance_miss = int((result["distance_miss"] & mask).sum())
    if correct + wrong_wrap + distance_miss != included:
        raise ValueError("frozen-denominator categories do not sum to the denominator")

    def percentage(value):
        return 100.0 * value / included if included else float("nan")

    return {
        "included": included,
        "correct": correct,
        "wrong_hop": wrong_wrap + distance_miss,
        "wrong_wrap": wrong_wrap,
        "distance_miss": distance_miss,
        "correct_pct": percentage(correct),
        "wrong_hop_pct": percentage(wrong_wrap + distance_miss),
        "wrong_wrap_pct": percentage(wrong_wrap),
        "distance_miss_pct": percentage(distance_miss),
    }


def pitch_bin_rows(result, mask, bins=PITCH_BINS):
    """Return miss arithmetic by local-gap bin on a fixed scorable mask."""
    mask = np.asarray(mask, dtype=bool)
    rows = []
    for lower, upper in bins:
        selected = mask & (result["gap"] >= lower) & (result["gap"] < upper)
        count = int(selected.sum())
        misses = int((selected & ~result["correct"]).sum())
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "misses": misses,
                "miss_pct": 100.0 * misses / count if count else float("nan"),
            }
        )
    return rows


def main():
    ref = load_reference(REF)
    if not os.path.exists(NORMALS):
        raise FileNotFoundError(
            f"missing {NORMALS}; rebuild it with build_reference_band.py before running this evidence"
        )
    normals = np.load(NORMALS)
    print(f"factor = 4.8/7.91 = {FACTOR:.4f}\n")
    for name, e in (("front", 1), ("back", -1)):
        p = load_grid(os.path.join(NATIVE, f"seed_segment_{name}"))
        d = p - ref.seed

        raw = score_prediction(ref, p, e)
        print(f"[{name}] 1. CONTROL unscaled   : {pct(summarize_score(raw))}"
              f"   <- must equal the published row")

        cal = score_prediction(ref, rescale_prediction(ref.seed, p, FACTOR), e)
        print(f"[{name}] 2. scaled by 4.8/7.91 : {pct(summarize_score(cal))}")
        frozen = summarize_on_mask(cal, raw["ok"])
        print(f"[{name}] 2b. original denominator: {pct(frozen)}")

        plateau = []
        for f in (0.50, 0.55, 0.607, 0.65, 0.70):
            rescored = score_prediction(ref, rescale_prediction(ref.seed, p, f), e)
            s = summarize_on_mask(rescored, raw["ok"])
            plateau.append(f"{f:.3f}:{s['correct_pct']:.1f}%")
        print(f"[{name}] 3. plateau            : {'  '.join(plateau)}")

        ws = summarize_score(score_prediction(ref, rescale_prediction(ref.seed, p, FACTOR), -e))
        print(f"[{name}] 4a. wrong-side control: correct {ws['correct_pct']:.2f}%  (must be ~0)")

        mag = np.linalg.norm(d * FACTOR, axis=-1)
        med = float(np.median(mag))
        unit = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        rng = np.random.default_rng(0)
        flat = unit.reshape(-1, 3)
        null_pred = ref.seed + flat[rng.permutation(len(flat))].reshape(unit.shape) * med
        ns = summarize_score(score_prediction(ref, null_pred, e))
        print(f"[{name}] 4b. within-window direction permutation: correct "
              f"{ns['correct_pct']:.2f}% at |d|={med:.1f}  <- low-curvature saturation")

        # The sign follows the bundle's documented normal convention. The
        # 9-voxel magnitude was chosen after inspecting this reference band's
        # 8.8/11.2-voxel median gaps, so this is a post-hoc saturation
        # diagnostic, not a pre-registered or deployable no-network baseline.
        sign = -1.0 if name == "front" else +1.0
        normal9 = summarize_score(
            score_prediction(ref, ref.seed + normals * (sign * 9.0), e)
        )
        print(f"[{name}] 4c. reference-pitch normal step: correct "
              f"{normal9['correct_pct']:.2f}% (|d|=9, sign {sign:+.0f})  <- post-hoc diagnostic")

        print(f"[{name}] 5. miss rate by local gap on original denominator:")
        for row in pitch_bin_rows(cal, raw["ok"]):
            upper = "inf" if np.isinf(row["upper"]) else f"{row['upper']:.0f}"
            print(
                f"  gap [{row['lower']:.0f},{upper}) vox: "
                f"{row['misses']}/{row['count']} = {row['miss_pct']:.2f}%"
            )
        print()


if __name__ == "__main__":
    main()
