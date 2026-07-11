# Local multi-wrap benchmark for compressed-region surface tracing

The Vesuvius Challenge community has identified surface tracing through
compressed/deformed scroll regions as the main blocker between current
segmentations and a full unrolling. The official team ships a trained
neural displacement tracer (`scrollprize/copy_displacement_latest`) built
for exactly this: given a partial segment mesh, predict the surface of the
adjacent wrap. We benchmarked it, quantitatively, in one of the hardest
regions we could find, using a local multi-wrap reference derived from the
same public segmentation.

> Companion write-up covering both this benchmark and the related
> winding-assignment result:
> https://github.com/LimeGS/winding-field/blob/main/WRITEUP.md

For the benchmark-package checklist, start with `docs/BENCHMARK.md`.

## Current result (2026-07-11)

| Configuration | Front wrong-hop | Back wrong-hop | Wrong-*wrap* |
|---|---|---|---|
| As released: native crop (128×384×384) + TTA, RTX 3090, 7.91 µm volume | **63.2%** | **56.3%** | 0% |
| Same predictions, displacement × 4.8/7.91 (resolution calibration) | **5.9%** | **7.7%** | 0% / 4.0% |
| Post-hoc reference-pitch diagnostic: seed + 9 vox along the local surface normal | 4.2% | 4.8% | — |

Wrong-hop means either assigning the wrong adjacent-wrap identity or missing
the expected wrap's recorded surface by at least half the local gap
(~100 µm spacing in this window, CT-verified). We pre-registered a <30% gate: the tool as
released misses it; the unit-corrected variant passes it with margin.
Earlier reduced-spec runs and the full history live in `docs/NT_AUDIT.md`.

**Why the correction is legitimate.** The checkpoint is trained on 4.8 µm
voxels; this volume is 7.91 µm. Reading the released CLI source (the exact
`vesuvius==0.2.4` wheel this benchmark ran): the model's displacement
output is applied directly in input-volume voxels (`world + slot_*_disp`, only a
max-displacement clamp), `--volume-scale` selects the zarr pyramid level,
`--tifxyz-voxel-size-um` only stamps output metadata — no unit conversion
exists anywhere in `vesuvius/neural_tracing` and no resolution requirement
is documented. The dataloader constructs `(dz,dy,dx)` targets from nearest
indices in the training crop and the loss compares those vectors without a
physical-spacing term. Given the checkpoint's reported 4.8 µm training
pitch, interpreting the same vector on 7.91 µm voxels makes its physical
length 1.65× larger. The code audit, including the provenance boundary on
the 4.8 µm value, is in `docs/UPSTREAM_UNIT_AUDIT.md`. Checks, in order: the
unscaled rerun reproduces the published row exactly first; the factor is
physics, not a fit (0.50–0.65 scores 89.6–96.4%); wrong-side
scoring of the calibrated predictions still fails 100%.

**Saturation caveat — read before celebrating.** In this window, once the
magnitude is right, a coherent local normal is sufficient: permuting the
model's directions *within this same low-curvature window* scores 93–97%,
and the 9-voxel normal-step diagnostic in the table matches the calibrated
model. The 9-voxel value was selected after inspecting this band's reference
gaps, so it is a saturation diagnostic, not a pre-registered or deployable
no-network baseline. This window certifies the unit mismatch and its
one-line fix; it does **not** certify learned directional skill beyond a
local normal. Ranking that needs harder terrain: multi-hop chaining, high
curvature, and a pitch estimator fixed before evaluation.

Reproduce: `python resolution_calibration.py` (saved run:
`docs/evidence/resolution_calibration_20260711.log`).

**Upstream implication:** this checkpoint can produce physically mis-scaled
displacements when `infer_rowcol_triplet_wraps` is used on a volume whose
pitch differs from its training grid. A generic CLI should expose an
opt-in displacement scale (default `1.0`) and checkpoint training-pitch
metadata, or document the requirement; it should not hard-code 4.8 µm for
unrelated checkpoints.

## The benchmark

**Setup:** the official checkpoint, unmodified, run against a
200×200-point seed window of a real public segment (`20240618142020`,
PHerc 0332 / Scroll 3), asked to predict the surface one wrap forward and
one wrap back, in a genuinely ultra-compressed window.

**Local multi-wrap reference**, without any human annotation: the same
segment spirals ~3 revolutions through the extracted band, so its own
geometry records where the neighboring wraps physically sit. Checked four
ways before trusting it:

- **Self-tests, not assumptions:** the readout metric reconstructs the
  seed's own points at 0.000 vox error; three earlier metric versions
  failed this and were discarded (`docs/NT_AUDIT.md`, "Metric integrity
  first").
- **Independent volume-truth cross-check** (`gap_verify.py`): raw CT
  intensity peaks along the seed normals confirm the ~100 µm sheet spacing
  from image intensity alone, no segmentation involved.
- **Discrimination controls:** wrong-side scoring fails 99%+; a
  constant-offset control at uncalibrated magnitude scores 100% wrong. That
  historical control remains a valid magnitude rejection but no longer
  supports a model-versus-geometry claim. At corrected magnitudes this
  window saturates: it distinguishes side/sign, but not a learned direction
  field from a simple coherent normal.
- **A gate set before knowing the answer:** wrong-hop < 30%, written down
  before spending on the GPU rerun ($0.23), reported without moving the
  threshold.

It only grades where the segment recorded the neighbor (over half the test
points are excluded rather than guessed). It is a local geometric
reference, not annotated ground truth, and it cannot evaluate unmapped
hard cases.

## Repo layout

```
docs/NT_AUDIT.md            full technical account (all stages, all numbers)
docs/BENCHMARK.md           reproducible benchmark spec/checklist for other methods
docs/CHECKSUMS.txt          SHA-256 checksums for the reference/seed files
docs/UPSTREAM_UNIT_AUDIT.md exact wheel/current-main training + inference unit audit
docs/evidence/              saved evidence logs (incl. resolution calibration)
band_r1145_200_xyz.npz      200-row local multi-wrap reference band — NOT in git
                            (48 MB, gitignored); regenerate bit-exactly with
                            build_reference_band.py before anything that reads it
winding_audit_v4.py          the headline audit: band construction, self-tests,
                              KD-tree wrap assignment, hop-correct metric
benchmark_core.py           shared KD-tree / scoring utilities
resolution_calibration.py   resolution-calibration evidence: control, rescore,
                              plateau, wrong-side + null + dumb-baseline controls
threshold_sweep.py           fixed-distance and gap-fraction threshold sweeps
make_ct_panels.py            CT cutaway panel generator for representative cells
gap_verify.py                independent CT-intensity cross-check of wrap spacing
sanity_check.py              output validity checks (finite, shape, plausibility)
run_inference.py             driver for the reduced-spec run (Mac, 128^3, no TTA)
gate_3090/run_native.py      driver for the gated rerun (RTX 3090, native + TTA)
gate_3090/score_native.py    scores a front/back output against the reference band
seed_segment/                the exact 200x200 tifxyz seed used throughout
out_mps/, gate_3090/out_native/   predicted front/back surfaces from each run
figures/                     displacement heatmaps, failure maps, CT panels
```

## Reproducing

```bash
pip install -r requirements.txt   # pulls the official vesuvius==0.2.4 package
```

The checkpoint (`scrollprize/copy_displacement_latest`, ~1.6 GiB, public,
non-gated on HuggingFace) is not bundled. The full local-reference band
(`band_r1145_200_xyz.npz`, 48 MB) is **not in git either** — a fresh clone
does not contain it. Rebuild it bit-exactly from the public segment PPM
first (~247 MB of HTTP range requests, ~2 min):

```bash
python build_reference_band.py --out band_r1145_200_xyz.npz --normals-out seed_normals.npy
shasum -a 256 -c docs/CHECKSUMS.txt   # FAILS if you skip the rebuild
```

Then score the bundled predictions and reproduce the headline table:

```bash
python resolution_calibration.py
python threshold_sweep.py --label native --out threshold_sweep_native.csv
python -m unittest discover -s tests
```

`seed_segment/` and the front/back output meshes ARE committed, so you can
inspect what the pipeline produced without rerunning inference.

## Limitations / what this does NOT claim

- **Resolution: from untested confound to identified mechanism.** The
  calibration finding establishes that the dominant *benchmark miss* at
  7.91 µm is a unit mismatch, correctable here by ×4.8/7.91. True native-resolution
  inference remains untested (no public chunk-addressable volume finer
  than 7.91 µm exists for PHerc 0332), and per the saturation caveat this
  window cannot rank calibrated-model direction quality against dumb
  geometry.
- **n = 1 window, one segment.** Blind-chosen, which happened to land
  ultra-compressed — representative of exactly the regime that matters,
  but one data point.
- The local multi-wrap reference is a public segmentation of unstated
  provenance, validated by internal self-consistency and an independent CT
  cross-check, not by a human-annotated label.
- Scorable cells require the neighboring wrap to exist inside the reference
  band. That likely selects easier geometry: it makes the as-released miss a
  conservative negative, but the calibrated 5.9/7.7% rows optimistic for
  general tracing performance.
- This benchmarks one checkpoint in one failure regime. It says nothing
  about performance in well-separated, uncompressed regions, where it may
  well be the right tool.

## License

MIT (code). Scan data referenced is CC BY-NC 4.0 from the Vesuvius
Challenge; the benchmarked checkpoint is public/non-gated on HuggingFace
under its own license (`scrollprize/copy_displacement_latest`).
