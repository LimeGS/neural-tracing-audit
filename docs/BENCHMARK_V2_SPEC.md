# Benchmark v2 — pre-registered specification (DRAFT v2.0.0-rc1)

Status: **DRAFT (rc1)** — parameters marked [TUNABLE] may be adjusted using
the tuning zone only (§2), then this file is frozen by a commit titled
`spec-freeze: v2.0.0` **before any evaluation inference runs**. After the
freeze, any change is an `AMENDMENT:` commit with a written justification;
results must report against the frozen spec plus amendment log.

## 0. Purpose and scope

Benchmark v1 (this repo) certified a unit-mismatch fix for
`copy_displacement_latest` on 7.91 um data, but its single low-curvature
window saturates once hop magnitude is right: a coherent local normal
matches the calibrated model, so v1 **cannot rank the model against plain
geometry**. v2 exists to make that ranking measurable, three additions:

1. a **pitch estimator fixed before evaluation** (no oracle magnitudes),
2. **curvature-stratified windows** selected by a deterministic rule,
3. **chained hops** with a physics-based per-hop verifier (v2.1).

v2 does NOT claim: native-resolution (4.8 um) performance, cross-scroll
generalization, or anything about uncompressed regions. All evaluation data
comes from one segment of one scroll (plus one replication window from a
second segment if budget allows, §9).

## 1. Data and frames

- Reference band: `band_r1145_200_xyz.npz` (PPM rows 1145-1344 of public
  segment `20240618142020`, PHerc 0332), rebuilt bit-exactly by
  `build_reference_band.py`; SHA-256 in `docs/CHECKSUMS.txt`.
- Volume: public `53keV_7.91um_Scroll3.zarr`, level 0.
- Winding classes and KD-trees: exactly as v1 (`winding_audit_v4.py`).
- Unit conversion constant for all model arms: 4.8/7.91 (fixed physics
  constant per `docs/UPSTREAM_UNIT_AUDIT.md`; not tunable).

## 2. Tuning zone (leakage control)

- TUNING ZONE: band columns **2,000-4,000**, all 200 rows.
- EVALUATION-ELIGIBLE COLUMNS: 0-25,705 excluding [1,500, 4,500] (tuning
  buffer) and excluding [12,550, 13,150] (v1 window + buffer; v1's window is
  previously-seen data and is reported only as a legacy exploratory row).
- Honest limitation, declared: rows cannot be made disjoint (the band is
  200 rows tall), and the spec authors have seen v1 results from this band.
  Tuning-zone discipline bounds, but does not eliminate, design leakage.
- Only parameters marked [TUNABLE] may be fitted in the tuning zone.

## 3. Curvature metric and window selection (deterministic)

- Candidate windows: 200 rows x 200 columns, column start on a stride-100
  grid over evaluation-eligible columns.
- Curvature proxy per column c: kappa(c) = median over rows r of
  `arccos( n(r, c+50) . n(r, c-50) )`, with n = unit seed normals from
  `build_reference_band.py`; window curvature = median kappa over its
  columns. Units: radians per 100 columns.
- Coverage per window: fraction of cells scorable for BOTH front and back
  single hops under v1 exclusion rules. Eligibility floor: coverage >= 0.40.
- Selection: among eligible windows, pick 2 lowest-kappa, 2 nearest the
  median kappa, 2 highest-kappa; enforce >= 300 columns separation between
  selected starts (on conflict, keep the higher-priority stratum pick and
  take the next candidate by rule).
- The selection runs once, by a committed script, and the resulting window
  list is committed as `docs/evidence/windows_v2.json` BEFORE any inference.
- Per-window validity gates (all must pass, failures excluded WITH REPORT,
  never silently replaced):
  a. self-test: seed reconstructs its own wrap at median 0.000 vox, p90 <= 0.5;
  b. wrong-side control: <= 5% correct;
  c. CT cross-check: ratio of CT median spacing to KD-gap median in
     [0.60, 1.60];
  d. saturation classification (§6): computed and reported either way.

## 4. Pitch estimators (fixed rules, no oracle)

Common CT profile recipe (from `gap_verify.py`): sample the public zarr by
trilinear interpolation along the seed normal, +/-40 vox at 1-vox steps;
Gaussian-smooth the profile, sigma = 2 samples [TUNABLE]; peaks =
`scipy.signal.find_peaks` with prominence >= 0.15 x (p95 - p5 of the
profile) [TUNABLE]; spacing = median of consecutive peak distances; a cell
ABSTAINS if it has < 2 accepted peaks.

- **P2 (primary, per-window scalar):** evaluate the recipe on a 9x9 cell
  grid across the window; window pitch = median over non-abstaining cells;
  the window's estimator ABSTAINS if < 30% of grid cells are valid.
- **P1 (secondary, per-cell):** cell pitch = median over the valid spacings
  of the 5x5 neighboring cells; abstain if < 5 valid neighbors; abstaining
  cells fall back to P2's window scalar, and the fallback fraction is
  reported.

Abstentions are excluded from estimator-arm denominators and reported as a
coverage table. Estimator recipes and parameters are frozen at spec freeze.

## 5. Arms

For every selected window and direction (front/back), score:

- **A. model-units:** released predictions x 4.8/7.91 (magnitude and
  direction from the model). v1's headline arm, for continuity.
- **B. model-direction + estimator-magnitude:** unit vector of the model
  displacement, length from P2 (primary) / P1 (secondary).
- **C. normal-direction + estimator-magnitude:** the bundle seed normal,
  same estimator magnitudes as B. Sign convention fixed a priori from v1's
  measurement: front = -normal, back = +normal (carried over as a constant,
  noted as learned in v1).
- **D. normal + 9 vox:** v1's post-hoc diagnostic, exploratory only.

The B-vs-C contrast with identical magnitudes isolates directional skill.
Scoring rule, thresholds and exclusions are v1's, unchanged
(`benchmark_core.py`), all rates on per-window frozen denominators.

## 6. Saturation classification and primary endpoint E1

- Per window, run the within-window direction-permutation null (seeded
  rng(0), magnitudes = per-cell |d| of arm B). A window is DISCRIMINATING
  if the null scores <= 85% correct; otherwise SATURATED (reported, and
  excluded from E1).
- **E1 (direction skill):** mean over discriminating windows of
  [correct%(B) - correct%(C)] using P2, plus a per-window sign count.
- **Gate G1 (declared verdict rule):** "evidence of directional skill" only
  if E1 >= +5.0 points AND B > C in at least 4 of 6 windows. Otherwise the
  published verdict is "no evidence of directional skill beyond a coherent
  local normal in this band". Both outcomes are published.
- Power pre-check (before selection is frozen): a synthetic
  curved-sheet simulation must show that arm C degrades below 85% at the
  curvature of the top stratum; if it does not, the top stratum cannot
  discriminate and the spec must be amended BEFORE freeze (documented), or
  E1's verdict language downgraded to "not measurable in this band".

## 7. Chained hops (v2.1 — separate freeze, same discipline)

- Chain arms: chain-A (model x 4.8/7.91 per hop) and chain-C (normal + P2
  per hop, normals recomputed from the predicted surface each hop).
  10 hops per direction from each selected window, using the upstream CLI's
  iterate mode; per-hop mesh sanity via `sanity_check.py`.
- **Per-hop verifier (physics, model-independent):** along the straight
  segment from hop origin to predicted point, count accepted CT peaks
  strictly between them (same profile recipe); a hop VERIFIES iff exactly
  one sheet boundary is crossed and the destination lies within +/-2 vox of
  a peak; low-SNR profiles ABSTAIN.
- **E2 (survival):** fraction of chains alive at k=5 (death = first
  verified-wrong hop), chain-A/B vs chain-C, reported twice: verified-only,
  and a pessimistic bound counting every abstained hop as death. Gate G2:
  >= +15 points on BOTH accountings; disagreement between accountings =>
  verdict "indeterminate".
- Protocol control: chaining the reference's own recorded surfaces (the
  <=2 hops the band supports) must survive >= 95%; below that the protocol
  is declared broken and no chain results are published.

## 8. Multiple-comparisons discipline

E1 and E2 are the only confirmatory endpoints. Everything else (per-bin
tables, P1 variants, arm D, legacy v1 window, per-window breakdowns) is
labeled exploratory and cannot support a headline claim.

## 9. Budget, order, replication

- v2.0 = §2-§6: tuning + selection + ~12 native-config inferences
  (6 windows x 2 directions; ~40 min RTX 3090, well under $1).
- v2.1 = §7: chain runs (~1-2 GPU hours).
- Stretch (budget permitting): one replication window on segment
  `20240716140050` with its own band build; reported separately, never
  pooled into E1/E2.

## 10. Reporting

Published regardless of outcome: frozen spec + amendment log, windows list,
all logs checksummed under `docs/evidence/`, coverage/abstention tables,
per-window control results, and raw per-cell scores sufficient to recompute
every headline number. Deviations discovered after publication are
corrected in-place with a dated correction note, following the practice
already used in this repo.
