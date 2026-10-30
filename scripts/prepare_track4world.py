"""Prepare a pinned public Track4World source tree with the MagicW0 feature patch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, destination, expected):
    if digest(archive) != expected:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    destination.mkdir()

    def safe_source(member, path):
        if "__pycache__" in Path(member.name).parts or member.name.endswith(".pyc"):
            return None
        return tarfile.data_filter(member, path)

    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter=safe_source)
    children = list(destination.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        raise ValueError("Expected one source directory in the archive")
    return children[0]


def verify(output, manifest):
    for name, hashes in manifest["files"].items():
        if not (output / name).is_file() or digest(output / name) != hashes["patched"]:
            raise ValueError(
                f"Teacher source differs from the pinned patched version: {name}"
            )
    if not (output / "utils3d/__init__.py").is_file():
        raise FileNotFoundError("Missing pinned utils3d dependency")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "third_party/Track4World"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded pinned Track4World tarball",
    )
    parser.add_argument("--utils3d-archive", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "third_party/track4world.json").read_text())
    output = args.output_dir.resolve()
    if args.check:
        verify(output, manifest)
        print(f"Verified patched Track4World source: {output}")
        return
    if output.exists():
        marker = output / ".magic-w0-source.json"
        if not marker.is_file() or json.loads(marker.read_text()) != manifest:
            raise FileExistsError(
                f"Refusing to overwrite an existing teacher tree: {output}; choose a new --output-dir"
            )
        verify(output, manifest)
        print(f"Already prepared: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".prepare-track4world-", dir=output.parent
    ) as directory:
        staging = Path(directory)
        archives = []
        specs = [
            (
                args.archive,
                f"https://codeload.github.com/TencentARC/Track4World/tar.gz/{manifest['revision']}",
                manifest["archive_sha256"],
            ),
            (
                args.utils3d_archive,
                f"https://codeload.github.com/jiah-cloud/utils3d/tar.gz/{manifest['utils3d_revision']}",
                manifest["utils3d_archive_sha256"],
            ),
        ]
        for index, (provided, url, checksum) in enumerate(specs):
            archive = provided or staging / f"source-{index}.tar.gz"
            if provided is None:
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    archive.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
            archives.append(unpack(archive, staging / f"tree-{index}", checksum))
        teacher, utils = archives
        patch = ROOT / "third_party/track4world-magicw0.patch"
        for name, hashes in manifest["files"].items():
            if digest(teacher / name) != hashes["upstream"]:
                raise ValueError(f"Unexpected upstream teacher file: {name}")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=teacher, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=teacher, check=True)
        shutil.copytree(utils, teacher / "utils3d")
        verify(teacher, manifest)
        (teacher / ".magic-w0-source.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        teacher.rename(output)
    print(f"Prepared Track4World feature teacher: {output}")


if __name__ == "__main__":
    main()
