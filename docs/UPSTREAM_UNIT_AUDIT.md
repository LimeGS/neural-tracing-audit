# Upstream displacement-unit audit

This note separates two claims that are easy to conflate:

1. **Code-derived:** the training target and inference output are expressed
   in array-grid voxels; the released CLI does not convert displacement
   vectors between physical voxel pitches.
2. **Provenance-derived:** `copy_displacement_latest` was reported as trained
   on 4.8 um data. That pitch is not encoded in the checkpoint configuration
   printed by the released loader (`model_config is empty`) and the
   Hugging Face repository currently has no model card. Preserve an upstream
   author/source confirmation before treating 4.8 um as self-describing model
   metadata or hard-coding it in a generic CLI.

## Exact release inspected

- Package: `vesuvius==0.2.4`, PyPI wheel
- Wheel SHA-256:
  `d251c81b7b5e0bc1cc91589e9fdb6d795f71c4898161fa3b80e9fe52d27539a1`
- Inference entrypoint:
  `vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py`
- Current-main comparison: `ScrollPrize/villa` commit
  `fed6a4cfef797dbc48c58617442b8d9d53f4aa02`; the implementation has changed
  substantially, but the same physical-unit conversion is still absent.

## Training chain

The relevant path in the 0.2.4 wheel is:

1. `datasets/dataset_rowcol_cond.py:EdtSegDataset.__init__` retargets segment
   coordinates only by the selected zarr pyramid factor (`2 ** volume_scale`).
   It does not normalize coordinates by microns per voxel.
2. Triplet surfaces are voxelized into the crop grid. Augmentations are
   applied to the volume and masks, then `create_neighbor_targets` recomputes
   targets from those transformed masks. In this wheel, the continuous
   scaling block in `models/augmentation/pipelines/training_transforms.py`
   is commented out; the active spatial transforms are discrete rotations /
   axis operations, which preserve voxel magnitudes.
3. `_compute_dense_displacement_field` runs a Euclidean distance transform
   and subtracts the crop's integer `(z, y, x)` axes from nearest-surface
   indices. The resulting target is explicitly `(dz, dy, dx)` in crop voxels.
4. `loss/displacement_losses.py:dense_displacement_loss` compares the raw
   predicted vector and raw target (`error = pred_field - gt_displacement`);
   Huber thresholds are also documented in voxels. No physical spacing is
   passed into the loss.

Therefore the model is supervised in the voxel units of its training array
level. This conclusion is direct from code and does not depend on benchmark
behavior.

## Inference chain

In the exact wheel's `infer_rowcol_triplet_wraps.py`, sampled branch vectors
are applied as `world + slot_*_disp`. `--volume-scale` retargets coordinates
and selects a zarr pyramid level. `--tifxyz-voxel-size-um` is passed to
`save_tifxyz` as output metadata; it does not rescale the vector. Current
`villa` main retains the same raw vector addition and metadata-only pitch.

The safe generic upstream design is therefore an explicit, opt-in
`--displacement-scale` with default `1.0`, plus checkpoint metadata that
records the training array pitch. Automatically using `4.8 / volume_pitch`
is appropriate only after the checkpoint's 4.8 um provenance is confirmed;
other checkpoints may use another pitch, a pyramid level, or multi-scale
training.

## Empirical check in this repository

`resolution_calibration.py` first reproduces the unscaled published row,
then applies `4.8 / 7.91` to the already-produced displacement vectors. The
factor sits on a broad performance plateau and passes the pre-registered
gate. This supports the 4.8 um provenance claim for this checkpoint/window,
but does not replace upstream metadata and does not establish directional
skill, as detailed in the saturation controls.
