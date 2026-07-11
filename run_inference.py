"""Reproduce the reduced-spec 128^3, no-TTA Mac inference configuration."""

import argparse
from pathlib import Path

from inference_driver import DEFAULT_VOLUME, build_argv, resolve_checkpoint, run_tracer


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=str(ROOT / "seed_segment"))
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--checkpoint", default=str(ROOT / "copy_displacement_latest.pth"))
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default=str(ROOT / "reproduced/out_mps"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args.checkpoint, args.download_checkpoint)
    tracer_argv = build_argv(
        tifxyz=args.seed,
        volume=args.volume,
        checkpoint=checkpoint,
        device=args.device,
        crop_size=(128, 128, 128),
        workers=1,
        out_dir=args.out,
        no_tta=True,
    )
    run_tracer(tracer_argv)


if __name__ == "__main__":
    main()
