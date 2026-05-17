#!/usr/bin/env python3
"""
Run electrode/channel ablations by retraining the existing CNN+LSTM recipe on
selected channel subsets.

The study answers two related questions:
- Leave-one-out: how much does performance drop when one electrode is removed?
- Single-channel sufficiency: how much information can each electrode carry alone?

Generated run directories are intentionally kept outside the curated project
bundle by default. Commit summaries, not model weights, unless a result is
selected for publication.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_NPZ = (
    "Projects/2-M16/subjects/2-M16/sessions/"
    "combined_20260319_081200_pruned_rest_events_0_1_2/processed/eeg_windows.npz"
)
DEFAULT_TRAIN_CONFIG = (
    "Projects/2-M16/subjects/2-M16/winning_model/model_run/train_config.json"
)
DEFAULT_OUT_DIR = "ablation_runs/channel_importance_2m16"


@dataclass(frozen=True)
class ChannelSubset:
    subset_id: str
    kind: str
    indices: tuple[int, ...]
    names: tuple[str, ...]
    omitted_channel: str = ""

    @property
    def channel_label(self) -> str:
        return "+".join(self.names)


def _safe_id(value: str) -> str:
    safe = []
    for ch in str(value):
        safe.append(ch if ch.isalnum() else "_")
    return "_".join("".join(safe).strip("_").split("__")) or "subset"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload.get("settings", payload) if isinstance(payload, dict) else {}


def _channel_names_from_npz(npz_path: Path) -> list[str]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "channel_names" in data:
            names = [str(v) for v in np.asarray(data["channel_names"]).reshape(-1)]
        else:
            x = np.asarray(data["X"])
            if x.ndim != 3:
                raise ValueError(f"Expected X to be 3D in {npz_path}, got {x.shape}")
            names = [f"ch{i + 1}" for i in range(int(x.shape[-1]))]
    if not names:
        raise ValueError(f"No channel names found in {npz_path}")
    return names


def _parse_modes(value: str) -> set[str]:
    aliases = {
        "single": "singles",
        "singles": "singles",
        "leave-one-out": "leave-one-out",
        "leave_one_out": "leave-one-out",
        "loo": "leave-one-out",
        "pairs": "pairs",
        "pair": "pairs",
        "all": "all",
        "all-nonempty": "all-nonempty",
        "all_nonempty": "all-nonempty",
    }
    modes: set[str] = set()
    for raw in value.replace(";", ",").split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token not in aliases:
            raise ValueError(f"Unknown subset mode: {raw}")
        modes.add(aliases[token])
    return modes


def _resolve_explicit_subset(token: str, channel_names: Sequence[str]) -> tuple[int, ...]:
    lookup = {name.upper(): idx for idx, name in enumerate(channel_names)}
    pieces = [p.strip() for p in token.replace(",", "+").split("+") if p.strip()]
    if not pieces:
        raise ValueError("Explicit subset token was empty")
    indices: list[int] = []
    for piece in pieces:
        key = piece.upper()
        if key.isdigit():
            idx = int(key)
        elif key in lookup:
            idx = lookup[key]
        else:
            raise ValueError(f"Unknown channel in explicit subset '{token}': {piece}")
        if idx < 0 or idx >= len(channel_names):
            raise ValueError(f"Channel index out of range in subset '{token}': {idx}")
        if idx not in indices:
            indices.append(idx)
    return tuple(indices)


def build_subset_plan(
    channel_names: Sequence[str],
    *,
    modes: Iterable[str],
    explicit_subsets: str = "",
) -> list[ChannelSubset]:
    c = len(channel_names)
    seen: set[tuple[int, ...]] = set()
    subsets: list[ChannelSubset] = []

    def add(kind: str, indices: Sequence[int], omitted: str = "") -> None:
        idx_tuple = tuple(int(i) for i in indices)
        if not idx_tuple or idx_tuple in seen:
            return
        seen.add(idx_tuple)
        names = tuple(str(channel_names[i]) for i in idx_tuple)
        if kind == "all":
            subset_id = "all"
        elif kind == "single":
            subset_id = f"single_{_safe_id(names[0])}"
        elif kind == "leave_one_out":
            subset_id = f"drop_{_safe_id(omitted)}"
        elif kind == "pair":
            subset_id = f"pair_{_safe_id('_'.join(names))}"
        else:
            subset_id = f"custom_{_safe_id('_'.join(names))}"
        subsets.append(
            ChannelSubset(
                subset_id=subset_id,
                kind=kind,
                indices=idx_tuple,
                names=names,
                omitted_channel=str(omitted),
            )
        )

    modes_set = set(modes)
    if "all-nonempty" in modes_set:
        for size in range(1, c + 1):
            for combo in itertools.combinations(range(c), size):
                add("all" if size == c else f"{size}_channel", combo)
    else:
        if "all" in modes_set:
            add("all", range(c))
        if "singles" in modes_set:
            for idx in range(c):
                add("single", [idx])
        if "pairs" in modes_set:
            for combo in itertools.combinations(range(c), 2):
                add("pair", combo)
        if "leave-one-out" in modes_set:
            for drop_idx in range(c):
                keep = [idx for idx in range(c) if idx != drop_idx]
                add("leave_one_out", keep, omitted=str(channel_names[drop_idx]))

    if explicit_subsets:
        for raw in explicit_subsets.split(";"):
            token = raw.strip()
            if token:
                add("custom", _resolve_explicit_subset(token, channel_names))

    return subsets


def write_subset_npz(source_npz: Path, output_npz: Path, subset: ChannelSubset) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_npz, allow_pickle=True) as data:
        payload: dict[str, Any] = {}
        channel_count = len(np.asarray(data["channel_names"]).reshape(-1)) if "channel_names" in data else None
        for key in data.files:
            value = data[key]
            if key == "X":
                x = np.asarray(value)
                if x.ndim != 3:
                    raise ValueError(f"Expected X to be 3D, got {x.shape}")
                if channel_count is not None and x.shape[-1] == channel_count:
                    payload[key] = x[:, :, list(subset.indices)]
                elif channel_count is not None and x.shape[1] == channel_count:
                    payload[key] = x[:, list(subset.indices), :]
                else:
                    payload[key] = x[:, :, list(subset.indices)]
            elif key == "channel_names":
                payload[key] = np.asarray(subset.names, dtype="U")
            else:
                payload[key] = np.asarray(value)
        payload["channel_ablation_source_npz"] = np.asarray(str(source_npz), dtype="U")
        payload["channel_ablation_subset_id"] = np.asarray(subset.subset_id, dtype="U")
        payload["channel_ablation_kind"] = np.asarray(subset.kind, dtype="U")
        payload["channel_ablation_indices"] = np.asarray(subset.indices, dtype=np.int64)
        payload["channel_ablation_names"] = np.asarray(subset.names, dtype="U")
        payload["channel_ablation_omitted_channel"] = np.asarray(subset.omitted_channel, dtype="U")
    np.savez_compressed(output_npz, **payload)


def _bool_flag(cmd: list[str], enabled: bool, true_flag: str, false_flag: str) -> None:
    cmd.append(true_flag if bool(enabled) else false_flag)


def _csvish(values: Any) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return values
    if isinstance(values, (list, tuple)):
        return ",".join(str(v) for v in values)
    return json.dumps(values)


def build_train_command(
    *,
    python_exe: str,
    train_script: Path,
    train_config: dict[str, Any],
    subset_npz: Path,
    run_dir: Path,
    seed: int,
    epochs: int | None,
    device: str,
) -> list[str]:
    lr = train_config.get("lr", train_config.get("learning_rate", 0.001))
    cmd = [
        python_exe,
        str(train_script),
        "--npz",
        str(subset_npz),
        "--run-dir",
        str(run_dir),
        "--subject-id",
        str(train_config.get("subject_id_filter") or train_config.get("subject_id") or ""),
        "--seed",
        str(seed),
        "--epochs",
        str(int(epochs if epochs is not None else train_config.get("epochs", 60))),
        "--batch-size",
        str(int(train_config.get("batch_size", 64))),
        "--lr",
        str(float(lr)),
        "--device",
        str(device),
        "--loss-action-weight",
        str(float(train_config.get("loss_action_weight", 1.0))),
        "--rest-weight",
        str(float(train_config.get("rest_weight", 1.0))),
        "--rest-balance-mode",
        str(train_config.get("rest_balance_mode", "core_event_equalized")),
        "--window-preprocess",
        str(train_config.get("window_preprocess", "center_detrend")),
        "--applicability-loss-weight",
        str(float(train_config.get("applicability_loss_weight", 0.5))),
        "--threshold-applicability",
        str(float(train_config.get("threshold_applicability", 0.5))),
        "--test-size",
        str(float(train_config.get("test_size", 0.2))),
        "--calibration-size",
        str(float(train_config.get("calibration_size", 0.1))),
        "--split-mode",
        str(train_config.get("split_mode", "group_trial")),
        "--aux-rest-session-policy",
        str(train_config.get("aux_rest_session_policy", "auto_train_only")),
        "--purge-seconds",
        str(float(train_config.get("purge_seconds", 0.0))),
        "--window-idx-leak-threshold",
        str(float(train_config.get("window_idx_leak_threshold", 0.65))),
        "--save-model",
        "finger_action_model.pt",
        "--save-scaler",
        "scaler.npz",
        "--save-preds",
        "test_predictions.npz",
        "--save-temperature",
        "temperature_scaling.json",
    ]
    if train_config.get("hop_seconds") is not None:
        cmd.extend(["--hop-seconds", str(float(train_config["hop_seconds"]))])
    if bool(train_config.get("strict_leakage", False)):
        cmd.append("--strict-leakage")
    if bool(train_config.get("non_rest_only", False)):
        cmd.append("--non-rest-only")
    _bool_flag(
        cmd,
        bool(train_config.get("active_finger_head", True)),
        "--active-finger-head",
        "--no-active-finger-head",
    )
    _bool_flag(
        cmd,
        bool(train_config.get("finger_applicability_head", True)),
        "--finger-applicability-head",
        "--no-finger-applicability-head",
    )
    action_weights = _csvish(train_config.get("action_weights"))
    if action_weights:
        cmd.extend(["--action-weights", action_weights])
    finger_weights = _csvish(train_config.get("finger_weights"))
    if finger_weights:
        cmd.extend(["--finger-weights", finger_weights])
    cmd.extend(["--rest-finger-loss-weight", str(float(train_config.get("rest_finger_loss_weight", 0.0)))])
    return cmd


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {"status": "missing_metrics", "run_dir": str(run_dir)}
    payload = json.loads(metrics_path.read_text())
    test = payload.get("test", {}) if isinstance(payload.get("test"), dict) else {}
    train = payload.get("train", {}) if isinstance(payload.get("train"), dict) else {}
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "action_acc": test.get("action_acc"),
        "finger_acc_non_rest": test.get("finger_acc_non_rest"),
        "n_test": test.get("n_test"),
        "n_test_non_rest": test.get("n_test_non_rest"),
        "train_action_acc": train.get("action_acc"),
        "train_finger_acc": train.get("finger_acc"),
    }


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.2f}"
    except Exception:
        return "n/a"


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_csv = out_dir / "summary.csv"
    fieldnames = [
        "subset_id",
        "kind",
        "channels",
        "omitted_channel",
        "n_channels",
        "seed",
        "status",
        "action_acc",
        "finger_acc_non_rest",
        "n_test",
        "n_test_non_rest",
        "action_drop_vs_all",
        "finger_drop_vs_all",
        "run_dir",
        "npz_path",
    ]

    baselines = {
        int(row["seed"]): row
        for row in rows
        if row.get("subset_id") == "all" and row.get("status") == "ok"
    }
    for row in rows:
        baseline = baselines.get(int(row["seed"]))
        if baseline and row.get("status") == "ok":
            for metric, out_key in (
                ("action_acc", "action_drop_vs_all"),
                ("finger_acc_non_rest", "finger_drop_vs_all"),
            ):
                if baseline.get(metric) is not None and row.get(metric) is not None:
                    row[out_key] = float(baseline[metric]) - float(row[metric])
                else:
                    row[out_key] = None
        else:
            row["action_drop_vs_all"] = None
            row["finger_drop_vs_all"] = None

    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ranked_action = sorted(
        ok_rows,
        key=lambda r: float(r.get("action_acc") or -1.0),
        reverse=True,
    )
    leave_one = [
        r for r in ok_rows if r.get("kind") == "leave_one_out" and r.get("action_drop_vs_all") is not None
    ]
    leave_one.sort(key=lambda r: float(r.get("action_drop_vs_all") or 0.0), reverse=True)
    singles = [r for r in ok_rows if r.get("kind") == "single"]
    singles.sort(key=lambda r: float(r.get("action_acc") or -1.0), reverse=True)

    lines = [
        "# Channel Ablation Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Completed runs: {len(ok_rows)}/{len(rows)}",
        "",
        "Interpretation: leave-one-out drops estimate necessity; single-channel accuracy estimates sufficiency. Both should be repeated across seeds before being treated as electrode importance.",
        "",
        "## Top Subsets By Action Accuracy",
        "",
        "| Rank | Subset | Channels | Seed | Action Acc (%) | Finger Acc Non-REST (%) |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ranked_action[:10], start=1):
        lines.append(
            f"| {rank} | {row['subset_id']} | {row['channels']} | {row['seed']} | "
            f"{_format_pct(row.get('action_acc'))} | {_format_pct(row.get('finger_acc_non_rest'))} |"
        )
    lines.extend([
        "",
        "## Leave-One-Out Importance",
        "",
        "| Omitted Electrode | Kept Channels | Seed | Action Drop vs All (pp) | Finger Drop vs All (pp) |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in leave_one:
        action_drop = row.get("action_drop_vs_all")
        finger_drop = row.get("finger_drop_vs_all")
        lines.append(
            f"| {row.get('omitted_channel', '')} | {row['channels']} | {row['seed']} | "
            f"{float(action_drop) * 100.0:.2f} | "
            f"{float(finger_drop) * 100.0:.2f} |"
        )
    lines.extend([
        "",
        "## Single-Channel Sufficiency",
        "",
        "| Electrode | Seed | Action Acc (%) | Finger Acc Non-REST (%) |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in singles:
        lines.append(
            f"| {row['channels']} | {row['seed']} | "
            f"{_format_pct(row.get('action_acc'))} | {_format_pct(row.get('finger_acc_non_rest'))} |"
        )
    lines.extend([
        "",
        "## Caveats",
        "",
        "- These are retraining ablations, not post-hoc occlusion tests.",
        "- A single seed can confound channel importance with optimizer variance.",
        "- Overlapping windows keep the same temporal-dependence limitations as the main manuscript.",
        "- If a subset looks strong offline, run pseudo-live replay before making command-path claims.",
        "",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines))


def run_sweep(args: argparse.Namespace) -> int:
    source_npz = Path(args.npz).expanduser()
    if not source_npz.is_absolute():
        source_npz = (REPO_ROOT / source_npz).resolve()
    train_config_path = Path(args.train_config).expanduser()
    if not train_config_path.is_absolute():
        train_config_path = (REPO_ROOT / train_config_path).resolve()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_config = _load_json(train_config_path)
    channel_names = _channel_names_from_npz(source_npz)
    modes = _parse_modes(args.subset_mode)
    subsets = build_subset_plan(
        channel_names,
        modes=modes,
        explicit_subsets=str(args.subsets or ""),
    )
    if args.max_subsets is not None:
        subsets = subsets[: max(0, int(args.max_subsets))]
    if not subsets:
        raise SystemExit("No channel subsets selected.")

    seeds = [
        int(token.strip())
        for token in str(args.seeds or train_config.get("seed", 43)).split(",")
        if token.strip()
    ]
    epochs = int(args.epochs) if args.epochs is not None else None
    train_script = REPO_ROOT / "2_train_model.py"

    manifest = {
        "schema_version": 1,
        "source_npz": str(source_npz),
        "train_config": str(train_config_path),
        "channel_names": channel_names,
        "subset_mode": args.subset_mode,
        "explicit_subsets": args.subsets,
        "seeds": seeds,
        "epochs": epochs if epochs is not None else train_config.get("epochs", 60),
        "subsets": [
            {
                "subset_id": s.subset_id,
                "kind": s.kind,
                "indices": list(s.indices),
                "names": list(s.names),
                "omitted_channel": s.omitted_channel,
            }
            for s in subsets
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows: list[dict[str, Any]] = []
    total = len(subsets) * len(seeds)
    count = 0
    for subset in subsets:
        subset_npz = out_dir / "datasets" / subset.subset_id / "eeg_windows.npz"
        if args.dry_run:
            print(f"[dry-run] subset {subset.subset_id}: {subset.channel_label}")
        else:
            if args.force or not subset_npz.exists():
                write_subset_npz(source_npz, subset_npz, subset)

        for seed in seeds:
            count += 1
            run_dir = out_dir / "runs" / subset.subset_id / f"seed_{seed}"
            cmd = build_train_command(
                python_exe=args.python,
                train_script=train_script,
                train_config=train_config,
                subset_npz=subset_npz,
                run_dir=run_dir,
                seed=seed,
                epochs=epochs,
                device=args.device,
            )
            print(f"[{count}/{total}] {subset.subset_id} seed={seed} channels={subset.channel_label}")
            if args.dry_run:
                print(" ".join(cmd))
                metrics = {"status": "dry_run", "run_dir": str(run_dir)}
            elif args.skip_existing and (run_dir / "metrics.json").exists():
                print(f"  skipping existing run: {run_dir}")
                metrics = _read_metrics(run_dir)
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "train.log"
                with log_path.open("w") as log_handle:
                    proc = subprocess.run(
                        cmd,
                        cwd=REPO_ROOT,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if proc.returncode != 0:
                    metrics = {"status": f"failed_{proc.returncode}", "run_dir": str(run_dir)}
                    print(f"  failed with exit code {proc.returncode}; see {log_path}")
                    if not args.keep_going:
                        rows.append(_row_for_subset(subset, seed, subset_npz, metrics))
                        write_summary(out_dir, rows)
                        return proc.returncode
                else:
                    metrics = _read_metrics(run_dir)
                    print(
                        "  action={}% finger_non_rest={}%".format(
                            _format_pct(metrics.get("action_acc")),
                            _format_pct(metrics.get("finger_acc_non_rest")),
                        )
                    )
            rows.append(_row_for_subset(subset, seed, subset_npz, metrics))
            write_summary(out_dir, rows)

    return 0


def _row_for_subset(
    subset: ChannelSubset,
    seed: int,
    subset_npz: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "subset_id": subset.subset_id,
        "kind": subset.kind,
        "channels": subset.channel_label,
        "omitted_channel": subset.omitted_channel,
        "n_channels": len(subset.indices),
        "seed": int(seed),
        "npz_path": str(subset_npz),
        **metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train channel-subset ablations for AlphaHand EEG electrode importance."
    )
    parser.add_argument("--npz", default=DEFAULT_SOURCE_NPZ, help="Source eeg_windows.npz.")
    parser.add_argument(
        "--train-config",
        default=DEFAULT_TRAIN_CONFIG,
        help="Base train_config.json or train.json recipe.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for ablation runs.")
    parser.add_argument(
        "--subset-mode",
        default="all,singles,leave-one-out",
        help="Comma list: all, singles, leave-one-out, pairs, all-nonempty.",
    )
    parser.add_argument(
        "--subsets",
        default="",
        help="Explicit semicolon-separated subsets, e.g. 'TP9+AF7;AF8+TP10'.",
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to train config seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for every run.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Training device.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch training.")
    parser.add_argument("--max-subsets", type=int, default=None, help="Limit subset count for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without writing datasets or training.")
    parser.add_argument("--force", action="store_true", help="Regenerate subset NPZ files even if present.")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs whose metrics.json already exists.",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed training run.")
    return parser


def main() -> int:
    return run_sweep(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
