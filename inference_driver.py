"""Shared reproducible driver for the official neural-tracing inference API."""

import hashlib
from pathlib import Path
import resource
import sys
import time
import traceback


DEFAULT_VOLUME = (
    "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/"
    "volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr"
)
CHECKPOINT_REPO = "scrollprize/copy_displacement_latest"
CHECKPOINT_FILENAME = "copy_displacement_latest.pth"
CHECKPOINT_SHA256 = "22cf4392f2f61e6a5548c7b68148e97fed4ee772abf4f842cc6b8d1ef3ca1370"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path, download=False):
    path = Path(path)
    if not path.exists() and download:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=CHECKPOINT_FILENAME,
            local_dir=path.parent,
        )
        path = Path(downloaded)
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. Re-run with --download-checkpoint "
            f"to fetch {CHECKPOINT_REPO}/{CHECKPOINT_FILENAME}."
        )
    observed = sha256_file(path)
    if observed != CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {observed}"
        )
    return path


def build_argv(
    tifxyz,
    volume,
    checkpoint,
    device,
    crop_size,
    workers,
    out_dir,
    no_tta,
):
    argv = [
        "--tifxyz-path", str(tifxyz),
        "--volume-path", str(volume),
        "--checkpoint-path", str(checkpoint),
        "--device", str(device),
        "--volume-scale", "0",
        "--crop-size", *(str(value) for value in crop_size),
        "--batch-size", "1",
        "--crop-input-workers", str(workers),
        "--bbox-band-workers", str(workers),
        "--out-dir", str(out_dir),
        "--verbose",
    ]
    if no_tta:
        argv.append("--no-tta")
    return argv


def peak_rss_gb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1e9 if sys.platform == "darwin" else 1024**2
    return value / divisor


def run_tracer(argv):
    from vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps import (
        parse_args,
        run,
    )

    print("[driver] argv:", argv, flush=True)
    args = parse_args(argv)
    started = time.time()
    try:
        outputs = run(args)
    except Exception:
        print("[driver] run() raised:", flush=True)
        traceback.print_exc()
        print(f"[driver] FAILED after {time.time() - started:.1f}s", flush=True)
        raise
    elapsed = time.time() - started
    print(
        f"[driver] run() completed in {elapsed:.1f}s, "
        f"peak RSS {peak_rss_gb():.2f} GB",
        flush=True,
    )
    print("[driver] outputs:", outputs, flush=True)
    return outputs
