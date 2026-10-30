"""Check an MagicW0 checkpoint's inference graph, tensor shapes and normalization."""

import argparse
from magic_w0.checkpoint import inference_payload
from magic_w0.training.dist_trainer.checkpoint_io import (
    load_checkpoint,
    resolve_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    payload = inference_payload(
        load_checkpoint(resolve_checkpoint(args.checkpoint), map_location="cpu")
    )
    print(f"Compatible inference layout: {len(payload['model'])} state entries")
    print(f"Normalization sources: {len(payload['normalization'])}")
    print(
        "VLM processor assets and GPU forward execution must also be checked before deployment."
    )


if __name__ == "__main__":
    main()
