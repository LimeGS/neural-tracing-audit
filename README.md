# Local multi-wrap benchmark for compressed-region surface tracing

The Vesuvius Challenge community has identified surface tracing through
compressed/deformed scroll regions as the main blocker between current
segmentations and a full unrolling. The official team ships a trained
neural displacement tracer (`scrollprize/copy_displacement_latest`) built
for exactly this: given a partial segment mesh, predict the surface of the
adjacent wrap. We benchmarked it, quantitatively, in one of the hardest
regions we could find, using a local multi-wrap reference derived from the
same public segmentation.

This repo is the harness, data, and current numbers. It is meant as a
community stress test for compressed-wrap failure modes, not as a claim
that neural tracing or winding methods are exhausted. The point is to make
future attempts comparable: same seed, same local reference, same scoring
rule, same numerator/denominator arithmetic.

For the benchmark-package checklist, start with `docs/BENCHMARK.md`.

## What we measured

**Setup:** the official checkpoint, unmodified, run against a
200×200-point seed window of a real public segment (`20240618142020`,
PHerc 0332 / Scroll 3), asked to predict the surface one wrap forward and
one wrap back. The seed window turned out to be genuinely ultra-compressed
— real inter-sheet spacing ~100 µm, confirmed independently from raw CT
intensity peaks along the surface normals, not just inferred from the
segmentation.

**Local multi-wrap reference**, without any human annotation: the same
segment spirals ~3 revolutions through the extracted band, so its own
geometry records where the neighboring wraps physically sit. We didn't take
that at face value — the readout metric went through four iterations, three
rejected by their own self-tests, before the final version (exact self-test:
0.000 vox error) was accepted. Full account, including the self-test
failures, in `docs/NT_AUDIT.md`.

**Result:**

| Configuration | Front wrong-hop | Back wrong-hop | Wrong-*wrap* component |
|---|---|---|---|
| Reduced spec (7.91 µm, 128³ crop, no TTA — Mac MPS) | 94.1% | 97.7% | 4.5% front / 7.6% back |
| Native crop (128×384×384) + TTA — RTX 3090, still 7.91 µm | **63.2%** | **56.3%** | **0%** |
| Constant-offset null (no network) | 100% | 100% | — |

Fixing crop size and TTA (2 of 3 known confounds — no finer public volume
exists for this scroll, so native *resolution* remains untested)
eliminates wrap-identity errors in this reference band, but the model still
lands points past half the true gap on most of the window. A
pre-registered gate (wrong-hop < 30%) was not met. That makes this
configuration a benchmark miss rather than a deployable compressed-region
tracer as tested; native-resolution and resolution-aware variants remain
open. For context, the corrected winding-field audit on this segment is
7,061 / 29,475 = 23.956% aggregate transition mismatch. That value is not
in the table because it is not a direct ranking: it measures disagreements
between adjacent sampled winding transitions, while this benchmark scores
predicted endpoints against a local multi-wrap reference.

## Why this is a real measurement

- **Self-tests, not assumptions.** The local-reference readout metric
  (`winding_audit_v4.py`) reconstructs the seed's own points through the
  full pipeline as a sanity check before trusting it on predictions —
  three earlier metric versions failed this and were discarded (see
  `docs/NT_AUDIT.md`, "Metric integrity first").
- **Independent volume-truth cross-check** (`gap_verify.py`): raw CT
  intensity peaks along the seed's surface normals confirm the same
  ~100 µm sheet spacing the local-reference band implies, from a completely
  different signal (image intensity, not segmentation geometry).
- **Two controls, not one number.** A constant-offset null (no network,
  just "seed + median displacement along the normal") scores 100% wrong at
  ~2× the distance of the real predictions — the network carries real
  signal. Scoring the front prediction against the *wrong-side* wrap fails
  99% — the metric discriminates direction cleanly.
- **A gate we set before knowing the answer.** Before spending on the RTX
  3090 rerun, we wrote down what result would make this variant compelling
  for compressed regions (wrong-hop < 30%). The result did not clear the
  bar; we report it as a benchmark miss rather than moving the threshold.
  Total cost of that decision: $0.23.

## Repo layout

```
docs/NT_AUDIT.md            full technical account (Stage 0-3, all numbers)
docs/BENCHMARK.md           reproducible benchmark spec/checklist for other methods
docs/CHECKSUMS.txt          SHA-256 checksums for the reference/seed files
band_r1145_200_xyz.npz      200-row local multi-wrap reference band — NOT in git
                            (48 MB, gitignored); regenerate bit-exactly with
                            build_reference_band.py before anything that reads it
winding_audit_v4.py          the headline audit: local-reference band construction,
                              self-tests, KD-tree wrap assignment, hop-correct metric
benchmark_core.py           shared KD-tree / scoring utilities
threshold_sweep.py           fixed-distance and gap-fraction threshold sweeps
threshold_sweep_*.csv        generated native/reduced sweep tables
make_ct_panels.py            CT cutaway panel generator for representative cells
figures/ct_panels/           generated pass/fail/excluded CT examples + manifest
gap_verify.py                independent CT-intensity cross-check of wrap spacing
sanity_check.py              output validity checks (finite, shape, plausibility)
run_inference.py             driver for the reduced-spec run (Mac, 128^3, no TTA)
gate_3090/run_native.py      driver for the gated rerun (RTX 3090, native crop + TTA)
gate_3090/score_native.py    scores a front/back output against the same reference band
seed_segment/                the exact 200x200 tifxyz seed used throughout (small, ~1.9MB)
out_mps/, gate_3090/out_native/   predicted front/back surfaces from each run
figures/                     displacement heatmaps + spatial failure maps
```

`run_inference.py` and `gate_3090/run_native.py` are runnable CLIs with
repo-local defaults (via `inference_driver.py`). The exact as-run originals
— which had session-specific hardcoded paths — are preserved at the initial
commit (`0bbbf5e`) for byte-level provenance; the rewrite reproduces every
headline number from the bundled artifacts (verified: native front
2,967/4,697 = 63.2% wrong-hop, back 4,895/8,697 = 56.3%, reduced 94.1/97.7%,
threshold-sweep CSVs byte-identical).

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
```

then verify everything (this step FAILS if you skip the rebuild):

```bash
shasum -a 256 -c docs/CHECKSUMS.txt
```

(The manual range-request recipe is also documented at the bottom of
`docs/NT_AUDIT.md`.) `seed_segment/` and the front/back output meshes ARE
committed, so you can inspect what the pipeline actually produced without
rerunning inference.

Generate threshold sweeps:

```bash
python threshold_sweep.py --label native --out threshold_sweep_native.csv
python threshold_sweep.py --label reduced \
  --front out_mps/seed_segment_front \
  --back out_mps/seed_segment_back \
  --out threshold_sweep_reduced.csv
```

Generate CT cutaway panels for representative pass/fail/excluded cells:

```bash
python make_ct_panels.py --out-dir figures/ct_panels --max-panels 10
```

The panel generator needs `zarr` and network access to the public CT zarr,
or a local zarr path passed with `--volume`.

Run the fast unit tests for the shared scoring and sweep utilities:

```bash
python -m unittest discover -s tests
```

Verify the bundled reference artifacts:

```bash
shasum -a 256 -c docs/CHECKSUMS.txt
```

## Limitations / what this does NOT claim

- **Resolution is the untested axis.** The rerun fixed crop size and TTA
  but not resolution — no chunk-addressable volume finer than 7.91 µm
  exists publicly for PHerc 0332 (the 3.24 µm scan ships only as raw
  per-slice TIFFs needing a nontrivial affine to this frame). "Fails at
  native resolution in compressed regions" is *not* established here —
  only "fails at 7.91 µm with native crop and TTA" is. Building that
  resampling adapter is real engineering work, not another GPU rental.
- **n = 1 window, one segment.** Blind-chosen (center of the extracted
  band), which happened to land ultra-compressed — representative of
  exactly the regime that matters, but one data point.
- The local multi-wrap reference is a public segmentation of unstated
  provenance (its own metadata carries no authorship field) — validated
  here by internal self-consistency and an independent CT cross-check, not
  by a human-annotated label.
- This benchmarks one checkpoint (`copy_displacement_latest`) in one
  failure regime. It says nothing about the model's performance in
  well-separated, uncompressed regions, where it may well be the right
  tool.

## License

MIT (code). Scan data referenced is CC BY-NC 4.0 from the Vesuvius
Challenge; the benchmarked checkpoint is public/non-gated on HuggingFace
under its own license (`scrollprize/copy_displacement_latest`).
