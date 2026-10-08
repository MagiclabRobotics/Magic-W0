"""Verify pinned downloads preserve an auditable asset receipt."""

import hashlib
import json
import sys
from types import ModuleType

from scripts import prepare_resources


def test_download_receipt_keeps_file_checksums(tmp_path, monkeypatch):
    manifest = json.loads(
        (prepare_resources.ROOT / "configs/resources.json").read_text()
    )["qwen_assets"]
    source = tmp_path / "snapshot"
    source.mkdir()
    for filename in manifest["files"]:
        (source / filename).write_text(filename)
    calls = []
    hub = ModuleType("huggingface_hub")

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    hub.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_resources", "--download-qwen", "--output-dir", str(output)],
    )
    prepare_resources.main()
    assert calls == [
        {
            "repo_id": manifest["repo_id"],
            "revision": manifest["revision"],
            "allow_patterns": manifest["files"],
        }
    ]
    receipt = json.loads((output / "resources.receipt.json").read_text())["qwen_assets"]
    assert receipt["repo_id"] == manifest["repo_id"]
    assert receipt["revision"] == manifest["revision"]
    assert receipt["files"] == {
        filename: hashlib.sha256((source / filename).read_bytes()).hexdigest()
        for filename in manifest["files"]
    }
