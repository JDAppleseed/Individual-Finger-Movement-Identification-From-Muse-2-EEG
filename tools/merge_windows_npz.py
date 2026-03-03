#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sequence_data import load_sequence_npz

DROP_SCALAR_MISMATCH_KEYS = {"features_path", "events_path"}


def _is_per_window(v, n: int) -> bool:
    return isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n


def _eq(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.shape == b.shape
            and np.all(a == b)
        )
    return a == b


def merge_npz(npz_paths: list[Path], out_path: Path) -> None:
    datasets = []
    for p in npz_paths:
        X, y_action, y_finger, meta = load_sequence_npz(p)
        datasets.append((X, y_action, y_finger, meta))

    shapes = [d[0].shape[1:] for d in datasets]
    if len(set(shapes)) != 1:
        raise SystemExit(f"Shape mismatch across datasets: {shapes}")

    X = np.concatenate([d[0] for d in datasets], axis=0)
    y_action = np.concatenate([d[1] for d in datasets], axis=0)
    y_finger = np.concatenate([d[2] for d in datasets], axis=0)

    metas = [d[3] for d in datasets]
    lengths = [len(d[1]) for d in datasets]

    merged_meta = {}
    all_keys = set().union(*(m.keys() for m in metas))

    for key in sorted(all_keys):
        values = [m.get(key) for m in metas]
        per_window_flags = [_is_per_window(v, n) for v, n in zip(values, lengths)]

        if key in DROP_SCALAR_MISMATCH_KEYS:
            base = next((v for v in values if v is not None), None)
            if base is None:
                continue
            if all((_eq(base, v) if v is not None else True) for v in values):
                merged_meta[key] = base
            else:
                print(
                    f"Skipping scalar meta key with per-session values: {key}"
                )
            continue

        if any(per_window_flags):
            if not all(per_window_flags):
                raise SystemExit(f"Per-window meta key missing in some datasets: {key}")
            merged_meta[key] = np.concatenate([np.asarray(v) for v in values], axis=0)
        else:
            base = next((v for v in values if v is not None), None)
            for v in values:
                if v is None:
                    continue
                if not _eq(base, v):
                    raise SystemExit(f"Scalar meta mismatch for key: {key}")
            merged_meta[key] = base

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y_action=y_action, y_finger=y_finger, **merged_meta)
    print(f"Wrote merged dataset: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge multiple eeg_windows.npz files into a single dataset."
    )
    p.add_argument(
        "--npz",
        action="append",
        required=True,
        help="Path to an eeg_windows.npz (repeatable)",
    )
    p.add_argument("--out", required=True, help="Output path for merged eeg_windows.npz")
    args = p.parse_args()

    npz_paths = [Path(x).expanduser().resolve() for x in args.npz]
    for pth in npz_paths:
        if not pth.exists():
            raise SystemExit(f"NPZ not found: {pth}")

    out_path = Path(args.out).expanduser().resolve()
    merge_npz(npz_paths, out_path)


if __name__ == "__main__":
    main()
