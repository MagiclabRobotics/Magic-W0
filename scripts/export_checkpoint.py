"""Export a full training checkpoint for the public MagicW0 inference runtime."""

from pathlib import Path
import argparse

from magic_w0.checkpoint import inference_payload
from magic_w0.training.dist_trainer.checkpoint_io import (
    load_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = resolve_checkpoint(args.input)
    if args.output.exists() or source.resolve() == args.output.resolve():
        raise FileExistsError("Export output must be a new file")
    payload = load_checkpoint(source, map_location="cpu", weights_only=True)
    save_checkpoint(args.output, inference_payload(payload))
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
