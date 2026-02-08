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
from datetime import datetime, timezone
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
        "--session-dir",
        type=str,
        default=None,
        help="Session directory to append to train/eval commands when missing.",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/sweep", help="Directory for sweep logs"
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not args.session_dir:
        if "--session-dir" not in args.train_cmd and "--npz" not in args.train_cmd:
            print("Session selection source: legacy_explicit")
            print(
                "❌ Missing --session-dir. Provide --session-dir or explicit --train-cmd with --npz."
            )
            return 2
        if "--session-dir" not in args.eval_cmd and "--npz" not in args.eval_cmd:
            print("Session selection source: legacy_explicit")
            print(
                "❌ Missing --session-dir. Provide --session-dir or explicit --eval-cmd with --npz."
            )
            return 2
        print("Session selection source: legacy_explicit")
    else:
        print("Session selection source: session_dir")
        session_dir_arg = shlex.quote(str(args.session_dir))
        if "--session-dir" not in args.train_cmd:
            args.train_cmd = f"{args.train_cmd} --session-dir {session_dir_arg}"
        if "--session-dir" not in args.eval_cmd:
            args.eval_cmd = f"{args.eval_cmd} --session-dir {session_dir_arg}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

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
