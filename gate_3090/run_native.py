"""Reproduce the RTX 3090 native-crop 128x384x384, TTA-on configuration."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_driver import (  # noqa: E402
    DEFAULT_VOLUME,
    build_argv,
    resolve_checkpoint,
    run_tracer,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=str(ROOT / "seed_segment"))
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--checkpoint", default=str(ROOT / "copy_displacement_latest.pth"))
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(ROOT / "reproduced/out_native"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args.checkpoint, args.download_checkpoint)
    tracer_argv = build_argv(
        tifxyz=args.seed,
        volume=args.volume,
        checkpoint=checkpoint,
        device=args.device,
        crop_size=(128, 384, 384),
        workers=4,
        out_dir=args.out,
        no_tta=False,
    )
    run_tracer(tracer_argv)


if __name__ == "__main__":
    main()
