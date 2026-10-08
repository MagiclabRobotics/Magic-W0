"""Prepare configuration assets and an explicitly selected checkpoint."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def validate_checkpoint(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty checkpoint: {path}")
    with path.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"Checkpoint is a Git LFS pointer, not model tensors: {path}"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qwen-source",
        type=Path,
        help="Copy assets from an existing Qwen3.5-2B directory",
    )
    p.add_argument(
        "--download-qwen",
        action="store_true",
        help="Download only config/tokenizer/processor assets",
    )
    p.add_argument(
        "--checkpoint-source", type=Path, help="Copy a trusted local checkpoint"
    )
    p.add_argument(
        "--checkpoint-repo",
        help="Download checkpoint from a chosen Hugging Face model repo",
    )
    p.add_argument("--checkpoint-file", default="magic_w0_robodojo.pt")
    p.add_argument(
        "--checkpoint-revision", help="Required immutable commit SHA when downloading"
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument(
        "--check",
        action="store_true",
        help="Only validate files, without loading 7GB tensors",
    )
    args = p.parse_args()
    if args.qwen_source and args.download_qwen:
        p.error("Choose --qwen-source or --download-qwen")
    if args.checkpoint_source and args.checkpoint_repo:
        p.error("Choose --checkpoint-source or --checkpoint-repo")
    if args.checkpoint_repo and (
        not args.checkpoint_revision
        or len(args.checkpoint_revision) != 40
        or any(c not in "0123456789abcdef" for c in args.checkpoint_revision)
    ):
        p.error(
            "Downloading a checkpoint requires --checkpoint-revision with a 40-character commit SHA"
        )
    if args.check and any(
        [
            args.qwen_source,
            args.download_qwen,
            args.checkpoint_source,
            args.checkpoint_repo,
        ]
    ):
        p.error("--check cannot be combined with preparation options")
    manifest = json.loads((ROOT / "configs/resources.json").read_text())
    output = args.output_dir.expanduser().resolve()
    assets = output / "qwen3.5-2b-assets"
    receipt = {}
    if args.qwen_source or args.download_qwen:
        source = args.qwen_source.expanduser().resolve() if args.qwen_source else None
        if source:
            missing = [
                file
                for file in manifest["qwen_assets"]["files"]
                if not (source / file).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Missing Qwen assets in {source}: {missing}")
        else:
            from huggingface_hub import snapshot_download

            spec = manifest["qwen_assets"]
            source = Path(
                snapshot_download(
                    repo_id=spec["repo_id"],
                    revision=spec["revision"],
                    allow_patterns=spec["files"],
                )
            )
        assets.mkdir(parents=True, exist_ok=True)
        for file in manifest["qwen_assets"]["files"]:
            if (source / file).resolve() != (assets / file).resolve():
                shutil.copy2(source / file, assets / file)
        receipt["qwen_assets"] = {
            "source": str(source),
            "files": {
                file: hashlib.sha256((assets / file).read_bytes()).hexdigest()
                for file in manifest["qwen_assets"]["files"]
            },
        }
        if args.download_qwen:
            receipt["qwen_assets"].update(
                {key: manifest["qwen_assets"][key] for key in ("repo_id", "revision")}
            )
    if args.checkpoint_source or args.checkpoint_repo:
        if args.checkpoint_repo:
            from huggingface_hub import hf_hub_download

            source = Path(
                hf_hub_download(
                    args.checkpoint_repo,
                    args.checkpoint_file,
                    revision=args.checkpoint_revision,
                )
            )
        else:
            source = args.checkpoint_source.expanduser().resolve()
        validate_checkpoint(source)
        output.mkdir(parents=True, exist_ok=True)
        target = output / "magic_w0_robodojo.pt"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        receipt["checkpoint"] = {
            "source": str(source),
            "repo_id": args.checkpoint_repo,
            "revision": args.checkpoint_revision,
            "size_bytes": target.stat().st_size,
        }
    if receipt:
        receipt_path = output / "resources.receipt.json"
        old = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        receipt_path.write_text(json.dumps({**old, **receipt}, indent=2) + "\n")
    if args.check:
        validate_checkpoint(output / "magic_w0_robodojo.pt")
        missing = [
            file
            for file in manifest["qwen_assets"]["files"]
            if not (assets / file).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing Qwen assets: {missing}")
        config = json.loads((assets / "config.json").read_text())
        if config.get("model_type") != "qwen3_5":
            raise ValueError(
                f"Expected Qwen3.5 assets, found model_type={config.get('model_type')!r}"
            )
        print(
            "Resource files present. Tensor/config compatibility is checked strictly when the policy starts."
        )
    elif not receipt:
        p.error("Select a preparation option or --check")


if __name__ == "__main__":
    main()
