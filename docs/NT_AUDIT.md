# Neural tracing (official `copy_displacement_latest`) — feasibility + winding audit

**2026-07-08.** Question: how does the official neural tracer behave in the
same compressed Scroll 3 regime studied by the winding-field experiments?
The winding comparison was corrected on 2026-07-10 after a sign-convention
bug was found in the winding-field solver scripts (offset propagation) —
not in this audit; the winding numbers changed because the fields changed,
plus a denominator re-aggregation: the corrected aggregate transition
mismatch is 7,061 / 29,475 = **23.956%** on segment `20240618142020` and
**21.525-26.637%** across methods 1-4 on the primary segment. Those rates
are context, not a direct ranking, because this document's neural metric
scores endpoints against a local multi-wrap reference rather than adjacent
transitions in a winding sequence.

Unless noted otherwise, file paths in this document are relative to the
repository root.

## Stage 0 — recon (verified against source, not memory)

- Module: `vesuvius/src/vesuvius/neural_tracing/` in ScrollPrize/villa.
- Two public, non-gated checkpoints on HF: `scrollprize/copy_displacement_latest`
  (1.58 GiB, updated 2026-07-01) and `scrollprize/extrap_displacement_latest`
  (1.14 GiB). Trained at 4.8 µm; native crop (128, 384, 384); 280.7M-param
  nnUNet-style net, 8 in / 6 out channels.
- Headless CLIs (`infer_rowcol_triplet_wraps`, `infer_global_extrap`) are
  device-configurable and run on this Mac (MPS + `PYTORCH_ENABLE_MPS_FALLBACK=1`);
  the VC3D GUI path (`trace_service.py`) is hard CUDA-only (cupy/cucim).
- PyPI `vesuvius==0.2.4` installs the working CLIs in ~80 s on macOS; current
  GitHub main does not (local volume-cartographer path dep + py3.14 pin).
- I/O model: existing partial tifxyz segment + OME-Zarr volume (chunk-on-demand,
  anonymous S3/dl.ash2txt) → `_front`/`_back` tifxyz (adjacent-wrap surfaces).
  Continuation tool — needs a seed mesh, no cold start.

## Stage 1 — smoke test (Mac, MPS)

Seed: 200×200 window of segment `20240618142020` (multi-wrap, 3.03 revs in
the 200-row band), rows 1145-1344 / cols 12750-12950, built from the PPM via
HTTP range requests (no tifxyz published for Scroll 3 → wrote one with
`vesuvius.tifxyz.write_tifxyz`, `compression=None`). Volume: 53keV 7.91 µm
standardized zarr, level 0. 128³ crops, no TTA, batch 1.

**Ran end-to-end: 34.8 s wall, 2.57 GB peak RSS, exit 0.** Outputs 100% valid,
zero non-finite; front/back split 94%/94% across the surface normal.

## Stage 2 — winding audit (the number that matters)

Metric: assign each predicted point to the nearest local-reference wrap (KD-tree per
winding class over the band's 5.07M points, winding classes from a
self-test-validated 2D phase unwrap; center-free). Hop-correct iff assigned
to the expected adjacent wrap and within half the local inter-wrap gap.
Self-test: seed vs own wrap = 0.000 vox. Umbilicus fitted from 25,376 PPM
normal lines at (1809.3, 1732.0) — the volume center is 130+ vox off and
makes radial assignment invalid (v1-v3 audits discarded via self-tests).

Window turns out ultra-compressed — CT intensity peaks along seed normals:
median inter-sheet spacing **13.0 vox = 103 µm** (p10-p90 8-22 vox),
independently confirming KD-tree reference gaps (8.8-11.2 vox). Both adjacent wraps
sit inside every 128³ crop, so crop size does NOT explain the failures.

n = 10,000 cells (every 2nd grid point):

| Case | Wrong-hop | Included / excluded (reference coverage) |
|---|---|---|
| NT front | **94.1%** | 4,613 / 5,387 |
| NT back | **97.7%** | 8,724 / 1,276 |
| Baseline A (constant median offset along normal) | 100.0% both | ~4.6k / ~8.6k |
| Control B (front vs wrong-side wrap) | 99.1% | 8,464 |
| Corrected winding field, same segment | 7,061 / 29,475 = 23.956% transition mismatch | segment-wide; different metric |

Distances to the correct wrap: NT front median 27.4 vox (216 µm), back
24.3 vox (192 µm) — vs local gaps of ~9-11 vox. Baseline A: 48.3/58.5 vox.
Stage 4 later showed that this uncalibrated baseline is useful only as a
gross-magnitude rejection: it does not establish learned directional skill
or a model-versus-geometry advantage.

**Failure mode:** NT predicts hops of median 34-41 vox along the normal
(~330-500 µm — the *canonical* wrap spacing) regardless of the actual local
~100 µm compression. Right side 94% of the time, wrong distance nearly
always: it lands 2-4 wraps away. It does not read local compression from
the volume in this regime.

## Caveats

- Run was out-of-spec vs training: 7.91 µm (native 4.8), ⅓-size crop, no
  TTA, single pass. All bias toward failure; none obviously explains
  predicting 3-4× the local spacing with both wraps visible in-crop.
- n=1 window, one segment — but blind-chosen and ultra-compressed, i.e.
  exactly the target regime where the winding field died.
- The local reference is an auto-segmentation; unwrap self-test + CT
  cross-check make gross winding errors here unlikely.
- Neural wrong-hop is not directly comparable to the corrected 23.956%
  winding transition-mismatch rate. The local band only covers wraps
  −2..+1, overshoots beyond ±1 are censored, and the two audits use
  different units and denominators. Each result should be read against its
  own definition and controls.

## Decision gate (pre-registered before any paid rerun)

Rerun this exact window at 4.8 µm-equivalent (3.24 µm Scroll 3 volume or
resampled), native (128,384,384) crop, TTA on, ~1 GPU-day on a 4090.
**Gate: wrong-hop < 30%** → this variant becomes compelling for compressed
regions; otherwise report the miss as a benchmark result and keep
native-resolution / resolution-aware variants as the next open test. This
gate was frozen using the then-reported 51.5% winding audit. The later sign
correction invalidated that number and its use as a comparative ceiling,
but did not retroactively change the neural gate.

## Stage 3 — gated rerun, RTX 3090, native crop + TTA (2026-07-08)

Rented vast.ai RTX 3090 ($0.1422/hr, ~98 min combined incl. one bad host,
**$0.23 total**), destroyed and confirmed after. Same seed, same reference band.
Fixed 2 of the 3 original confounds:

- **Crop size: native (128, 384, 384)** ✅ (was 128³ on the Mac)
- **TTA: on**, default mirror mode, 8 flip combos ✅ (was off on the Mac)
- **Resolution: still 7.91 µm**, NOT fixed — no finer/native zarr exists for
  Scroll 3 (`volumes_zarr_standardized/` has only the 7.91 µm entry; a
  3.24 µm volume exists but only as raw per-slice TIFFs with a nontrivial
  affine registration to the seed's frame, not zarr-loadable by the CLI as
  shipped — building that adapter was out of scope for a live paid-GPU run).

Wall time 155.6 s, confirmed `cuda` device (RTX 3090), peak GPU memory
22,852 / 24,576 MiB (matches the ~23 GB extrapolated from the Mac's 128³
measurement — no OOM, correctly predicted).

n = 10,000 cells, same metric/script (self-test 0.000 vox):

| Config | Front wrong-hop | Back wrong-hop |
|---|---|---|
| Mac reduced-spec (7.91µm, 128³, no TTA) | 94.1% | 97.7% |
| **3090 native crop + TTA (7.91µm)** | **63.2%** | **56.3%** |

For context, the corrected winding-field audit reports 23.956% aggregate
transition mismatch on this segment and 21.525-26.637% across methods 1-4
on the primary segment. It is not inserted into the endpoint table because
it is a different measurement with a different denominator.

Distances: front median 7.7 vox (61 µm) vs local gap 11.2 vox; back median
5.2 vox (41 µm) vs local gap 8.8 vox. **Wrong-*wrap* component dropped to
0% for both** (was the dominant failure mode on the Mac) — every remaining
failure now is "assigned the correct adjacent wrap, landed too far within
it," not "jumped to the wrong wrap entirely."

**Verdict against the pre-registered gate: NOT MET.** 63.2%/56.3% is far
above the frozen 30% bar. Per the discipline set before this experiment,
this configuration is a benchmark miss for the compressed-region use case
as tested. The corrected winding result cannot supply a head-to-head
ceiling because its transition metric is not the neural endpoint metric.
Fixing crop-size and TTA closed most of the gap (94→63%, 98→56%, and eliminated
wrong-wrap jumps outright), but this was a 2-of-3 rerun, not the full
3-of-3 pre-registered one — resolution remains a genuinely untested
variable, not a ruled-out one. Testing it would require building a custom
affine-resampling zarr adapter for the 3.24 µm volume, a real,
separately-scoped engineering task, not another GPU rental — not pursued
as part of this gate.

Blockers hit and resolved: first rented host had a broken Docker Hub path
(destroyed, re-rented); second host took ~35 min to leave "loading"; plain
`scp` stalled (switched to `rsync`); plain `curl`/`aria2c` against HF's Xet
CDN produced a byte-perfect-length but *corrupted* checkpoint (wrong
sha256) — fixed with `huggingface_hub.hf_hub_download` (Xet-aware),
verified sha256 `22cf4392f2f61e6a5548c7b68148e97fed4ee772abf4f842cc6b8d1ef3ca1370`.

## Artifacts

`docs/BENCHMARK.md` (community benchmark spec), `docs/CHECKSUMS.txt`,
`band_r1145_200_xyz.npz` (48 MB local-reference band — not in git,
regenerable bit-exactly via `build_reference_band.py`),
`benchmark_core.py` (shared scoring utilities), `winding_audit_v4.py`
(headline audit), `threshold_sweep.py` + `threshold_sweep_*.csv`
(gap-fraction/fixed-distance sweeps), `make_ct_panels.py` (representative
CT cutaway generator), `gap_verify.py` (CT volume-truth check),
`run_inference.py`, `sanity_check.py`, `run_mps.log`,
`winding_failure_map_v4.png`, `displacement_{front,back}.png`,
`seed_segment/` + `out_mps/` (tifxyz in/out).
`gate_3090/` — Stage 3 rerun: `run_native.py`, `run_native.log`,
`score_native.py`, `score_native.log`, `winding_failure_map_native.png`,
`out_native/` (front/back tifxyz, ~1MB).
Rebuild recipe for the band: HTTP range-request rows 1145-1344 of
`.../Scroll3/PHerc332.volpkg/paths/20240618142020/20240618142020.ppm`
(header 74 bytes, row-major float64 x,y,z,nx,ny,nz, width 25706).

---

## Stage 4 — Resolution-calibration finding (2026-07-11)

The Stage 3 native-config numbers (63.2/56.3% wrong-hop) measured the tool
**as released**. A later cross-project pass found the dominant cause of that
benchmark miss: the
checkpoint is reported as trained on 4.8 um voxels (provenance boundary in
docs/UPSTREAM_UNIT_AUDIT.md), the volume is 7.91 um, and the
released CLI applies displacement outputs directly in input-volume voxels —
no unit conversion exists anywhere in `vesuvius/neural_tracing` (verified on
the exact `vesuvius==0.2.4` wheel: `world + slot_*_disp` with only a
max-displacement clamp; `--volume-scale` = zarr pyramid level;
`--tifxyz-voxel-size-um` = output metadata only), and no resolution
requirement is documented. The dataloader constructs target vectors from
nearest-surface indices in the crop grid and the loss compares them without
a physical-spacing term. Given the checkpoint's reported 4.8 um training
pitch, the vector's physical interpretation on 7.91 um voxels is 1.65x
larger. See `docs/UPSTREAM_UNIT_AUDIT.md`, including the fact that 4.8 um is
upstream provenance rather than self-describing checkpoint metadata.

Multiplying displacements by 4.8/7.91 = 0.607 (nothing else):
front 63.17% -> **5.85%** wrong-hop, back 56.28% -> **7.71%** (back gains
4.0% wrong-wrap, previously 0). Controls: unscaled rerun reproduces the
published rows exactly; broad plateau 0.50-0.65 (89.6-96.4% correct, not a
fitted point); wrong-side
scoring of calibrated predictions fails 100%.

**Saturation caveat:** at corrected magnitudes this window stops
discriminating a learned direction field from a coherent local normal. A
within-window direction permutation scores 93-97%, and a 9-voxel
normal-step diagnostic scores 95.8/95.2%, matching the calibrated model
(94.1/92.3%). The 9-voxel magnitude was selected after inspecting this
reference band's gaps, so it is explicitly post-hoc, not a deployable or
pre-registered baseline. This window certifies the unit mismatch and its
fix, not learned directional skill; that requires multi-hop chaining,
curvature, and a pitch rule frozen before evaluation.

Evidence: `resolution_calibration.py` +
`docs/evidence/resolution_calibration_20260711.log`. The same evidence now
reports misses by gap bin: front is worst below 7 vox, while back is worst
above 16 vox; both are near-zero in their central pitch bands. This supports
local pitch estimation as the residual sub-problem without implying that a
deployable estimator has already been demonstrated.
