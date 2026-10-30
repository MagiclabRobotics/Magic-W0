import json
from pathlib import Path
import subprocess
import sys

import pytest

from eval.robodojo.bootstrap import bootstrap, REQUIRED
from eval.robodojo.paths import resolve_checkpoint
from eval.robodojo.run import (
    main,
    simulator_shim,
    SHIM_MARKER,
    wait_ready,
    stop_process,
    free_port,
)
from scripts.prepare_resources import validate_checkpoint


def test_dry_run_does_not_write_and_exposes_correct_commands(tmp_path, capsys):
    assert (
        main(
            [
                "--task",
                "stack_bowls",
                "--robodojo-root",
                str(tmp_path / "external"),
                "--policy-gpu",
                "2",
                "--env-gpu",
                "3",
                "--seed",
                "7",
                "--output-dir",
                str(tmp_path / "runs"),
                "--dry-run",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "eval.robodojo.server" in text
    assert "--seed 7" in text
    assert "--device_id 3" in text
    assert "CUDA_VISIBLE_DEVICES=2" in text
    assert list(tmp_path.iterdir()) == []


def test_simulator_shim_refuses_existing_policy(tmp_path):
    target = tmp_path / "XPolicyLab/policy/Magic_W0"
    target.mkdir(parents=True)
    (target / "model.py").write_text("existing code\n")
    with pytest.raises(FileExistsError):
        simulator_shim(tmp_path, install=True)
    assert (target / "model.py").read_text() == "existing code\n"


def test_owned_shim_is_reusable(tmp_path):
    target = simulator_shim(tmp_path, install=True)
    assert (target / "deploy.py").read_text().startswith(SHIM_MARKER)
    assert "eval_one_episode_batch" in (target / "deploy.py").read_text()
    assert simulator_shim(tmp_path, install=False) == target
    separate = simulator_shim(tmp_path, install=True, policy_name="Magic_W0_Local")
    assert separate.name == "Magic_W0_Local"


def test_readiness_and_process_cleanup(tmp_path):
    port = free_port("127.0.0.1")
    ready = tmp_path / "ready.json"
    code = f"""
import asyncio,json
from pathlib import Path
from websockets.asyncio.server import serve
async def handler(connection):
    await connection.wait_closed()
async def main():
    async with serve(handler, "127.0.0.1", {port}):
        Path({str(ready)!r}).write_text(json.dumps({{"run_id":"test", "url":"ws://127.0.0.1:{port}"}}))
        await asyncio.Future()
asyncio.run(main())
"""
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        wait_ready(process, "127.0.0.1", port, 5, ready_file=ready, run_id="test")
        assert process.poll() is None
    finally:
        stop_process(process)
    assert process.poll() is not None


def test_readiness_reports_early_server_failure(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(2)"], start_new_session=True
    )
    process.wait()
    with pytest.raises(RuntimeError, match="status 2"):
        wait_ready(
            process,
            "127.0.0.1",
            19000,
            5,
            ready_file=tmp_path / "ready.json",
            run_id="test",
        )


def test_external_benchmark_contract_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Incompatible"):
        bootstrap(tmp_path)
    for name in REQUIRED:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    bootstrap(tmp_path)


def test_checkpoint_selection_and_lfs_pointer_rejection(tmp_path):
    for step in [9, 20]:
        (tmp_path / f"checkpoint-epoch-1-step-{step}.pt").touch()
    assert resolve_checkpoint({"checkpoint_path": str(tmp_path)}).name.endswith(
        "step-20.pt"
    )
    pointer = tmp_path / "pointer.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:fake\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        validate_checkpoint(pointer)


def test_resource_receipt_and_validation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "configs/resources.json").read_text())
    source = tmp_path / "qwen"
    source.mkdir()
    for name in manifest["qwen_assets"]["files"]:
        (source / name).write_text("{}")
    (source / "config.json").write_text('{"model_type":"qwen3_5"}')
    checkpoint = tmp_path / "tiny.pt"
    checkpoint.write_bytes(b"test-only-file-not-model-tensors")
    output = tmp_path / "output"
    script = root / "scripts/prepare_resources.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--qwen-source",
            str(source),
            "--checkpoint-source",
            str(checkpoint),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(script), "--check", "--output-dir", str(output)],
        check=True,
    )
    receipt = json.loads((output / "resources.receipt.json").read_text())
    assert receipt["checkpoint"]["size_bytes"] == checkpoint.stat().st_size
    assert len(receipt["qwen_assets"]["files"]["config.json"]) == 64
