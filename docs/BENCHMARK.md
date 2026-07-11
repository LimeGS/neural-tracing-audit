# Local multi-wrap reference benchmark

This package turns the compressed-region neural-tracing audit into a
reproducible stress test. It is intended for ScrollFiesta, VC3D, neural
surface tracers, winding-number methods, bridge-cut / weld algorithms, and
any other method that predicts adjacent-wrap geometry.

The reference is deliberately called a **local multi-wrap reference**, not
ground truth. It is derived from one public segmentation and validated by
self-consistency plus an independent CT intensity check; it is not a
manual papyrological annotation.

Unless noted otherwise, file paths in this document are relative to the
repository root.

## Status

This is makeable now with the artifacts already in this repository.

- Already packaged: seed tifxyz (the 48 MB local-reference band NPZ is
  NOT in git — regenerate it bit-exactly with `build_reference_band.py`),
  reduced-spec and native-config predicted front/back tifxyz outputs,
  scoring code, run configs/logs, failure maps, CT-spacing verifier,
  threshold sweeps, generated CT panels, CT-panel generator, checksums,
  and the full technical note.
- Rebuildable if you want to verify the NPZ independently: HTTP
  range-request rows 1145-1344 of the source PPM; exact recipe is at the
  end of `docs/NT_AUDIT.md`.

## 1. Reference strip / crop

| Field | Value |
|---|---|
| Scroll / volume package | PHerc 0332 / Scroll 3, `PHerc332.volpkg` |
| CT volume | `volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr`, level 0 |
| Volume shape | z,y,x = 9778 x 3550 x 3400 |
| Voxel size | 7.91 um |
| Segment | `20240618142020` |
| Source geometry | `paths/20240618142020/20240618142020.ppm` |
| Reference band rows | 1145-1344 inclusive, 200 rows |
| Full band width | 25,706 columns |
| Seed crop columns | 12750-12949 inclusive, 200 columns |
| Seed tifxyz | `seed_segment/` |
| Scored grid | every 2nd seed cell: 100 x 100 = 10,000 cells per direction |
| Orientation | front = +1 winding class, back = -1 winding class, as assigned by the local unwrap |
| Preprocessing | no CT preprocessing for scoring; geometry is read as tifxyz / PPM coordinates in 7.91 um voxel space |

The seed crop is committed; the multi-wrap reference band is NOT in git —
reconstruct it first with `build_reference_band.py` (bit-exact from the
public PPM rows above), then verify everything with `docs/CHECKSUMS.txt`.

## 2. Local reference geometry

The 200-row band spans about 3.03 revolutions of the same segmented
surface. A 2D phase unwrap assigns every valid band point to a local
winding class relative to the seed center.

Native-config class populations from `gate_3090/score_native.log`:

| Class | Points |
|---|---:|
| -2 | 114,509 |
| -1 | 1,654,338 |
| 0 | 1,675,193 |
| +1 | 1,631,237 |

Classes +2/+3 are absent in this band. One KD-tree is built per populated
class. The seed cells all self-classify as class 0: 10,000 / 10,000.

## 3. Scoring logic

Reference implementation: `gate_3090/score_native.py`.

Pseudocode:

```python
for each scored seed cell s:
    expected = +1 for front, -1 for back
    gap = distance(seed[s], nearest_surface_point(class=expected))
    predicted = tracer_output[s]
    d_expected = distance(predicted, nearest_surface_point(class=expected))
    nearest_class = argmin_class distance(predicted, KDTree[class])

    if nearest expected point or local gap point is on band row 0 or 199:
        excluded
    elif seed cell is not class 0:
        excluded
    elif nearest_class != expected:
        wrong_adjacent_wrap_identity
    elif d_expected >= threshold:
        correct_wrap_but_distance_miss
    else:
        hop_correct
```

The headline threshold is `threshold = 0.5 * local_gap`.

## 4. Scorable versus excluded

A point is scorable when:

- its seed cell is in winding class 0;
- the expected adjacent class exists in the reference band;
- the nearest expected-wrap point used for the prediction distance is not
  on the first or last band row;
- the nearest expected-wrap point used for the local gap is not on the
  first or last band row.

The row-edge rule avoids counting cells whose true nearest adjacent-wrap
surface may lie just outside the 200-row reference band.

## 5. Thresholds and gap measurement

The local gap for each cell is the 3D KD-tree distance from the seed point
to the expected adjacent winding class. The strict headline criterion is
half of that cell's local gap.

Native-config measured gaps:

| Direction | Median local gap | p10 | p90 |
|---|---:|---:|---:|
| Front | 11.2 vox = 88 um | 8.4 | 14.4 |
| Back | 8.8 vox = 70 um | 6.6 | 11.5 |

Sensitivity already reported:

- Half-gap: headline benchmark.
- Full-gap: native-config numbers improve substantially, but in a ~100 um
  compressed spiral, landing past half-gap is already geometrically
  ambiguous.
- Fixed-distance alternatives and full-gap scoring are tabulated in
  `threshold_sweep_native.csv` and `threshold_sweep_reduced.csv`.

Independent CT check: `gap_verify.py` samples intensity profiles along
seed normals. Across 9 cells / 96 inter-peak spacings, median spacing is
13.0 vox = 103 um, p10-p90 8-22 vox, confirming the local reference gaps
are real compression rather than a segmentation-only artifact.

## 6. Tracer configs tested

| Config | Model | Volume | Crop | TTA | Hardware | Output |
|---|---|---|---|---|---|---|
| Reduced spec | `scrollprize/copy_displacement_latest` | 7.91 um zarr, level 0 | 128 x 128 x 128 | off | Apple M2 Pro, MPS | `out_mps/` |
| Native crop + TTA | same checkpoint, sha256 in `docs/NT_AUDIT.md` | 7.91 um zarr, level 0 | 128 x 384 x 384 | mirror, 8 flips | RTX 3090 | `gate_3090/out_native/` |

The checkpoint was public and non-gated on HuggingFace, 1.58 GiB, updated
2026-07-01. The released model is trained at 4.8 um; both runs here use the
available 7.91 um chunk-addressable zarr, so native-resolution performance
is still an open test.

## 7. Metrics

Counts are `numerator / scorable denominator`. Excluded cells are listed
separately.

| Config | Direction | Scorable / excluded | Hop-correct | Wrong-hop | Wrong adjacent-wrap identity | Correct wrap but distance miss | Median expected-wrap distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Reduced spec | Front | 4,613 / 5,387 | 274 / 4,613 = 5.9% | 4,339 / 4,613 = 94.1% | 4.5% | 89.6% | 27.4 vox = 216 um |
| Reduced spec | Back | 8,724 / 1,276 | 202 / 8,724 = 2.3% | 8,522 / 8,724 = 97.7% | 7.6% | 90.1% | 24.3 vox = 192 um |
| Native crop + TTA | Front | 4,697 / 5,303 | 1,730 / 4,697 = 36.8% | 2,967 / 4,697 = 63.2% | 0.0% | 63.2% | 7.7 vox = 61 um |
| Native crop + TTA | Back | 8,697 / 1,303 | 3,802 / 8,697 = 43.7% | 4,895 / 8,697 = 56.3% | 0.0% | 56.3% | 5.2 vox = 41 um |

Direction correctness:

- Reduced-spec run: front/back split is 94% / 94% across the surface normal
  (`run_mps.log`), but the distance is usually too large.
- Discrimination control: scoring the front prediction against the wrong
  side fails 99.1%, so the metric distinguishes side/sign. It does not by
  itself establish precise learned direction within the correct half-space.

Overshoot:

- Reduced spec predicts median normal hops of 34-41 vox, roughly
  330-500 um, while CT spacing is about 100 um. The technical note
  summarizes this as landing 2-4 wraps away in the compressed window.
- Native crop + TTA eliminates the wrong-wrap component in this reference
  band; the remaining miss is correct class but beyond half-gap.

## 8. Self-tests and null baselines

Self-test:

- Seed points scored against class 0 through the full KD-tree pipeline:
  median 0.000 vox, max 0.000 vox in `score_native.log`.
- All 10,000 scored seed cells classify as class 0.

Metric development history:

- v1 radial assignment was rejected because the volume center was more
  than 130 voxels from the fitted umbilicus.
- v2 z-matched assignment was rejected because reconstructing the seed's
  own wrap gave p90 error 61.7 vox.
- v3 phase unwrap passed; v4 removed axis-dependent distance and uses true
  3D nearest-surface distance.

Null / controls:

- Constant median-offset baseline along normals: 100% wrong-hop in both
  directions, at about 2x the network's distance. This historical control
  rejects a grossly wrong magnitude; it is not evidence that the network
  outperforms local geometry.
- Wrong-side scoring control: front scored against back expectation fails
  99.1%.
- CT intensity spacing check: median 13.0 vox = 103 um, independently
  matching the compressed local geometry.

Resolution-calibrated diagnostic (added after the pre-registered run):

- Scaling the same predictions by `4.8/7.91` gives 5.90% front and 7.73%
  back wrong-hop on the original scorable denominators. The exact training
  and inference unit chain is documented in `UPSTREAM_UNIT_AUDIT.md`.
- A within-window direction permutation remains 93-97% correct, and a
  post-hoc 9-voxel normal step reaches 95.8/95.2%. These are saturation
  diagnostics: this low-curvature window certifies the magnitude correction,
  but cannot rank learned direction against a coherent local normal.
- Residual miss rates vary sharply with local gap, but not monotonically on
  both sides. That supports testing a pre-specified local-pitch estimator;
  it does not yet establish pitch variation as the sole residual cause.
- Scorability requires recorded neighboring geometry. This likely favors
  easier regions, so the calibrated row may be optimistic for general tracing.

## 9. Visual examples

Packaged figures:

- `figures/winding_failure_map_native.png`: front on the left, back on the
  right. Green = hop-correct, orange = correct wrap but > half-gap, red =
  wrong adjacent-wrap identity, gray = excluded. This is the clearest
  pass/fail visualization for the native-config benchmark.
- `figures/winding_failure_map_v4.png`: same visualization for the
  reduced-spec run.
- `figures/displacement_front.png` and `figures/displacement_back.png`:
  displacement magnitude heatmaps.

CT cutaway generator:

- `figures/ct_panels/` contains generated oblique CT panels and a
  manifest for representative cells: pass, borderline half-gap miss,
  severe distance miss, wrong-wrap identity miss if present, and
  excluded-by-edge.
- `make_ct_panels.py` regenerates those panels from the same scoring
  logic.
- It requires `zarr` plus access to the public CT zarr or a local zarr copy.
  Install with `pip install -r requirements.txt`, then run:

```bash
python make_ct_panels.py --out-dir figures/ct_panels --max-panels 10
```

The script writes PNGs and `figures/ct_panels/manifest.csv`.

## 10. How to run a new method

1. Produce predicted front/back tifxyz grids for `seed_segment/`, with the
   same 200 x 200 lattice.
2. Put them in directories containing `x.tif`, `y.tif`, `z.tif`.
3. Run:

```bash
python gate_3090/score_native.py \
  --reference band_r1145_200_xyz.npz \
  --front path/to/front_tifxyz \
  --back path/to/back_tifxyz \
  --out my_failure_map.png
```

Threshold sweeps:

```bash
python threshold_sweep.py --label native --out threshold_sweep_native.csv
python threshold_sweep.py \
  --label reduced \
  --front out_mps/seed_segment_front \
  --back out_mps/seed_segment_back \
  --out threshold_sweep_reduced.csv
```

The bundled CSVs include gap-fraction thresholds (including half-gap and
full-gap), fixed-voxel thresholds, and fixed-micron thresholds.

## Checksums

`docs/CHECKSUMS.txt` includes SHA-256 checksums for the reference NPZ
(regenerated, see above) and the committed seed tifxyz coordinate files.
