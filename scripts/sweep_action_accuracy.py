#!/usr/bin/env python3
"""
Sweep training/evaluation to maximize overall action accuracy.

Runs Step 2 + Step 3 for a grid of hyperparameters and records metrics
from each eval_manifest.json into a CSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.session_layout import SessionLayout, resolve_latest_run_dir


def _parse_list(val: str, cast=float) -> List:
    items = []
    for chunk in val.replace(",", " ").split():
        if chunk.strip() == "":
            continue
        items.append(cast(chunk))
    return items


def _run(cmd: str) -> None:
    print(f"▶ {cmd}", flush=True)
    proc = subprocess.run(shlex.split(cmd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


def _load_metrics(session_dir: Path) -> tuple[str, dict]:
    run_dir = resolve_latest_run_dir(session_dir)
    if run_dir is None:
        raise RuntimeError("No run directory found after training.")
    layout = SessionLayout(session_dir)
    manifest_path = layout.reports_root / run_dir.name / "eval_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing eval manifest: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text())
    return run_dir.name, manifest.get("metrics", {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep action-accuracy settings.")
    parser.add_argument(
        "--session-dir",
        type=str,
        required=True,
        help="Session directory containing processed windows.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Projects/2-M16/subjects/2-M16/config/train.json",
        help="Train config path (used for defaults).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run train/eval (default: current interpreter).",
    )
    parser.add_argument("--rest-weights", type=str, default="0.2,0.5,1.0")
    parser.add_argument("--action-weights", type=str, default="1.0,2.0")
    parser.add_argument("--lrs", type=str, default="0.001")
    parser.add_argument("--epochs", type=str, default="60")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--resume-csv",
        type=str,
        default=None,
        help="Existing sweep CSV to resume/append (skips completed runs).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/sweep",
        help="Directory to write sweep CSV results.",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    if not session_dir.exists():
        raise SystemExit(f"Session dir not found: {session_dir}")

    rest_weights = _parse_list(args.rest_weights, float)
    action_weights = _parse_list(args.action_weights, float)
    lrs = _parse_list(args.lrs, float)
    epochs = _parse_list(args.epochs, int)
    seeds = _parse_list(args.seeds, int)

    fieldnames = [
        "run_id",
        "seed",
        "rest_weight",
        "action_weight",
        "lr",
        "epochs",
        "action_acc",
        "rest_tpr",
        "rest_fpr",
        "finger_acc_non_rest",
        "action_ece",
        "smoothed_action_acc",
        "smoothed_rest_tpr",
        "smoothed_rest_fpr",
    ]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_csv:
        csv_path = Path(args.resume_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"action_sweep_{timestamp}.csv"

    completed = set()
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                key = (
                    str(row.get("seed", "")).strip(),
                    str(row.get("rest_weight", "")).strip(),
                    str(row.get("action_weight", "")).strip(),
                    str(row.get("lr", "")).strip(),
                    str(row.get("epochs", "")).strip(),
                )
                completed.add(key)

    def _key(rest_w, action_w, lr, epochs_val, seed_val):
        return (
            str(int(seed_val)),
            f"{rest_w}",
            f"{action_w}",
            f"{lr}",
            str(int(epochs_val)),
        )

    grid = list(itertools.product(rest_weights, action_weights, lrs, epochs, seeds))
    total = len(grid)
    remaining = [g for g in grid if _key(*g) not in completed]
    print(
        f"Grid size: {total} | completed: {len(completed)} | remaining: {len(remaining)}",
        flush=True,
    )

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        idx = 0
        for rest_w, action_w, lr, epochs, seed in remaining:
            idx += 1
            print(
                f"\n=== Run {idx}/{len(remaining)} | rest_weight={rest_w} "
                f"action_weight={action_w} lr={lr} epochs={epochs} seed={seed} ===",
                flush=True,
            )

            train_cmd = (
                f"{shlex.quote(str(args.python))} 2_train_model.py"
                f" --config {shlex.quote(str(args.config))}"
                f" --session-dir {shlex.quote(str(session_dir))}"
                f" --rest-weight {rest_w}"
                f" --loss-action-weight {action_w}"
                f" --lr {lr}"
                f" --epochs {epochs}"
                f" --seed {seed}"
                f" --device {shlex.quote(str(args.device))}"
                f" --num-workers {int(args.num_workers)}"
                f"{' --pin-memory' if args.pin_memory else ''}"
            )
            _run(train_cmd)

            eval_cmd = (
                f"{shlex.quote(str(args.python))} 3_evaluate_model.py"
                f" --session-dir {shlex.quote(str(session_dir))}"
            )
            _run(eval_cmd)

            run_id, metrics = _load_metrics(session_dir)
            row = {
                "run_id": run_id,
                "seed": seed,
                "rest_weight": rest_w,
                "action_weight": action_w,
                "lr": lr,
                "epochs": epochs,
                "action_acc": metrics.get("action_acc"),
                "rest_tpr": metrics.get("rest_tpr"),
                "rest_fpr": metrics.get("rest_fpr"),
                "finger_acc_non_rest": metrics.get("finger_acc_non_rest"),
                "action_ece": metrics.get("action_ece"),
                "smoothed_action_acc": metrics.get("smoothed_action_acc"),
                "smoothed_rest_tpr": metrics.get("smoothed_rest_tpr"),
                "smoothed_rest_fpr": metrics.get("smoothed_rest_fpr"),
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"✅ {run_id}: action_acc={row['action_acc']}, rest_tpr={row['rest_tpr']}, "
                f"finger_acc_non_rest={row['finger_acc_non_rest']}",
                flush=True,
            )

    print(f"\nSweep complete. Results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
