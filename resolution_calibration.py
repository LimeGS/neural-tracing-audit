"""Resolution-calibration finding (2026-07-11): evidence generator.

The released checkpoint was trained on 4.8 um voxels; the public PHerc0332
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
     - BUT a saturation caveat: in this window, once the magnitude is right,
       nearly any direction lands within half-gap (permuted-direction null
       and a dumb normal-step baseline both score >90%). This window
       therefore certifies the magnitude fix, NOT the model's directional
       skill over dumb geometry; harder terrain (multi-hop chaining,
       curvature, variable gaps) is needed for that.

Run from the repository root:  python resolution_calibration.py
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_core import load_reference, load_grid, score_prediction, summarize_score

REF = os.path.join(HERE, "band_r1145_200_xyz.npz")
NATIVE = os.path.join(HERE, "gate_3090", "out_native")
NORMALS = os.path.join(HERE, "seed_normals.npy")
FACTOR = 4.8 / 7.91          # training voxel pitch / volume voxel pitch


def pct(s):
    return f"correct {s['correct_pct']:.2f}% wrong-hop {s['wrong_hop_pct']:.2f}% (incl {s['included']})"


def main():
    ref = load_reference(REF)
    normals = np.load(NORMALS) if os.path.exists(NORMALS) else None
    print(f"factor = 4.8/7.91 = {FACTOR:.4f}\n")
    for name, e in (("front", 1), ("back", -1)):
        p = load_grid(os.path.join(NATIVE, f"seed_segment_{name}"))
        d = p - ref.seed

        raw = score_prediction(ref, p, e)
        print(f"[{name}] 1. CONTROL unscaled   : {pct(summarize_score(raw))}"
              f"   <- must equal the published row")

        cal = score_prediction(ref, ref.seed + d * FACTOR, e)
        print(f"[{name}] 2. scaled by 4.8/7.91 : {pct(summarize_score(cal))}")

        plateau = []
        for f in (0.50, 0.55, 0.607, 0.65, 0.70):
            s = summarize_score(score_prediction(ref, ref.seed + d * f, e))
            plateau.append(f"{f:.3f}:{s['correct_pct']:.1f}%")
        print(f"[{name}] 3. plateau            : {'  '.join(plateau)}")

        ws = summarize_score(score_prediction(ref, ref.seed + d * FACTOR, -e))
        print(f"[{name}] 4a. wrong-side control: correct {ws['correct_pct']:.2f}%  (must be ~0)")

        mag = np.linalg.norm(d * FACTOR, axis=-1)
        med = float(np.median(mag))
        unit = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        rng = np.random.default_rng(0)
        flat = unit.reshape(-1, 3)
        null_pred = ref.seed + flat[rng.permutation(len(flat))].reshape(unit.shape) * med
        ns = summarize_score(score_prediction(ref, null_pred, e))
        print(f"[{name}] 4b. permuted-dir null : correct {ns['correct_pct']:.2f}% at |d|={med:.1f}"
              f"  <- saturation caveat")

        if normals is not None:
            best = None
            for mag9 in (9.0, 13.0):
                for sign in (+1.0, -1.0):
                    q = ref.seed + normals * (sign * mag9)
                    s = summarize_score(score_prediction(ref, q, e))
                    if best is None or s["correct_pct"] > best[0]:
                        best = (s["correct_pct"], mag9, sign)
            print(f"[{name}] 4c. dumb normal-step  : best correct {best[0]:.2f}% "
                  f"(|d|={best[1]:.0f}, sign {best[2]:+.0f})  <- what the model must beat\n")


if __name__ == "__main__":
    main()
