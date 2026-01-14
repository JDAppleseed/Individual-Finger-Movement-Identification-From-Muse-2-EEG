#!/usr/bin/env python3
"""
STEP 6 — Sweep runner (train + eval with separate logs)

Runs training and evaluation commands in sequence for a configurable number
of runs. Each run writes both a train log and an eval log.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run_and_log(cmd: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            shlex.split(cmd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sweep with separate train/eval logs."
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of sweep runs to execute"
    )
    parser.add_argument(
        "--train-cmd",
        type=str,
        default=f"{sys.executable} 2_train_model.py",
        help="Train command to execute",
    )
    parser.add_argument(
        "--eval-cmd",
        type=str,
        default=f"{sys.executable} 3_evaluate_model.py",
        help="Eval command to execute",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/sweep", help="Directory for sweep logs"
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for run_idx in range(1, max(1, args.runs) + 1):
        run_id = f"{timestamp}_run{run_idx}"
        train_log = log_dir / f"{run_id}_train.log"
        eval_log = log_dir / f"{run_id}_eval.log"
        _run_and_log(args.train_cmd, train_log)
        _run_and_log(args.eval_cmd, eval_log)
        print(f"✅ Completed run {run_idx}/{args.runs}: {train_log}, {eval_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
