"""Run explicit task/seed combinations; aggregate execution status, not metrics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from .paths import REPO_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/robodojo")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args()
    batch = args.output_dir.resolve() / (
        "batch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    results = []
    for task in args.tasks:
        for seed in args.seeds:
            cmd = [
                sys.executable,
                "-m",
                "eval.robodojo.run",
                "--task",
                task,
                "--seed",
                str(seed),
                "--output-dir",
                str(batch),
                *extra,
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            result = subprocess.run(cmd, cwd=REPO_ROOT)
            results.append({"task": task, "seed": seed, "exit_code": result.returncode})
    if not args.dry_run:
        batch.mkdir(parents=True, exist_ok=True)
        (batch / "batch-summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Batch execution summary: {batch / 'batch-summary.json'}")
    return int(any(row["exit_code"] for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
