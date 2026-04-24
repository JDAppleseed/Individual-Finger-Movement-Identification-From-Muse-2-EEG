#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_ROOT = REPO_ROOT / "Projects" / "2-M16" / "subjects" / "2-M16"
SESSIONS_ROOT = SUBJECT_ROOT / "sessions"

SOURCE_SESSIONS = (
    "2-M16_20260216_150056_01",
    "2-M16_20260317_190134",
    "2-M16_20260315_145838_01",
)
PRUNE_REST_EVENT_IDS: Mapping[str, set[int]] = {
    "2-M16_20260216_150056_01": {0, 1, 2},
}
DEFAULT_OUT = (
    SESSIONS_ROOT
    / "combined_20260319_081200_pruned_rest_events_0_1_2"
    / "processed"
    / "eeg_windows.npz"
)
PATH_PREFIXES = (
    str(REPO_ROOT) + "/",
    "/Users/oliverbuchanan/Desktop/ISEF/Individual-Finger-Movement-Identification-From-Muse-2-EEG/",
)
DROP_SCALAR_MISMATCH_KEYS = {"features_path", "events_path"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_text(value: str) -> str:
    out = str(value)
    for prefix in PATH_PREFIXES:
        out = out.replace(prefix, "")
    return out


def _sanitize_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind not in {"U", "S", "O"}:
        return arr
    if arr.dtype.kind == "S":
        arr = arr.astype("U")
    sanitized = np.asarray(arr).astype("U", copy=True)
    flat = sanitized.reshape(-1)
    for idx, value in enumerate(flat):
        flat[idx] = _repo_relative_text(str(value))
    return sanitized


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _is_per_window(arr: Any, n: int) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim >= 1 and len(arr) == n


def _filter_payload(
    payload: Mapping[str, np.ndarray], session_id: str
) -> Dict[str, np.ndarray]:
    y_action = np.asarray(payload["y_action"])
    n = int(len(y_action))
    keep = np.ones(n, dtype=bool)

    prune_ids = PRUNE_REST_EVENT_IDS.get(session_id)
    if prune_ids:
        if "event_id" not in payload:
            raise SystemExit(f"{session_id} is missing event_id metadata required for pruning")
        event_id = np.asarray(payload["event_id"], dtype=np.int64)
        prune_mask = np.isin(event_id, np.asarray(sorted(prune_ids), dtype=np.int64))
        prune_mask &= y_action.astype(np.int64) == 0
        keep &= ~prune_mask

    out: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if _is_per_window(arr, n):
            arr = arr[keep]
        out[key] = _sanitize_array(arr)
    return out


def _same_array(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    if a.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def build_reference(source_npzs: Sequence[Path]) -> Dict[str, np.ndarray]:
    payloads: List[Dict[str, np.ndarray]] = []
    lengths: List[int] = []

    for path in source_npzs:
        if not path.exists():
            raise SystemExit(f"Missing source NPZ: {path}")
        session_id = path.parents[1].name
        payload = _filter_payload(_load_npz(path), session_id=session_id)
        payloads.append(payload)
        lengths.append(int(len(payload["y_action"])))

    shapes = {tuple(payload["X"].shape[1:]) for payload in payloads}
    if len(shapes) != 1:
        raise SystemExit(f"Source window shape mismatch: {sorted(shapes)}")

    keys = set().union(*(payload.keys() for payload in payloads))
    ordered_keys = ["X", "y_action", "y_finger"] + sorted(keys - {"X", "y_action", "y_finger"})
    merged: Dict[str, np.ndarray] = {}

    for key in ordered_keys:
        values = [payload.get(key) for payload in payloads]
        per_window = [
            _is_per_window(value, n) if value is not None else False
            for value, n in zip(values, lengths)
        ]
        if any(per_window):
            if not all(per_window):
                raise SystemExit(f"Per-window metadata key missing in some sources: {key}")
            merged[key] = np.concatenate([np.asarray(value) for value in values], axis=0)
            continue

        first = next((np.asarray(value) for value in values if value is not None), None)
        if first is None:
            continue
        if key in DROP_SCALAR_MISMATCH_KEYS:
            if all(
                value is None or _same_array(first, np.asarray(value))
                for value in values
            ):
                merged[key] = first
            continue
        for value in values:
            if value is not None and not _same_array(first, np.asarray(value)):
                raise SystemExit(f"Scalar metadata mismatch for key: {key}")
        merged[key] = first

    return merged


def _summarize(payload: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    y_action = np.asarray(payload["y_action"], dtype=np.int64)
    y_finger = np.asarray(payload["y_finger"], dtype=np.int64)
    sessions = np.asarray(payload["session_id"]).astype("U") if "session_id" in payload else np.array([])

    def counts(arr: np.ndarray) -> Dict[str, int]:
        values, nums = np.unique(arr, return_counts=True)
        return {str(value): int(num) for value, num in zip(values.tolist(), nums.tolist())}

    return {
        "shape": list(np.asarray(payload["X"]).shape),
        "action_counts": counts(y_action),
        "finger_counts": counts(y_finger),
        "session_counts": counts(sessions) if sessions.size else {},
        "channel_names": [str(x) for x in np.asarray(payload.get("channel_names", [])).reshape(-1)],
    }


def _compare(expected: Mapping[str, np.ndarray], actual_path: Path) -> List[str]:
    if not actual_path.exists():
        return [f"Output NPZ is missing: {actual_path}"]
    actual = _load_npz(actual_path)
    errors: List[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"Missing key in output: {key}")
            continue
        if not _same_array(np.asarray(expected_value), np.asarray(actual[key])):
            errors.append(f"Mismatch for key: {key}")
    for key in actual:
        if key not in expected:
            errors.append(f"Unexpected key in output: {key}")
    return errors


def _source_paths(values: Iterable[str] | None) -> List[Path]:
    if values:
        return [Path(value).expanduser().resolve() for value in values]
    return [
        SESSIONS_ROOT / session_id / "processed" / "eeg_windows.npz"
        for session_id in SOURCE_SESSIONS
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the published 2-M16 reference dataset from the three "
            "source-session eeg_windows.npz files."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source session eeg_windows.npz path. Repeat to override the defaults.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT.relative_to(REPO_ROOT)),
        help="Output NPZ path. Defaults to the published combined 2-M16 dataset.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Rebuild in memory and compare against --out without writing files.",
    )
    args = parser.parse_args()

    sources = _source_paths(args.source)
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    payload = build_reference(sources)
    summary = _summarize(payload)

    expected_summary = {
        "shape": [12447, 64, 4],
        "action_counts": {"0": 2404, "1": 4814, "2": 5229},
        "finger_counts": {
            "0": 2404,
            "1": 2252,
            "2": 1742,
            "3": 2051,
            "4": 1922,
            "5": 2076,
        },
        "session_counts": {
            "2-M16_20260216_150056_01": 9744,
            "2-M16_20260315_145838_01": 1059,
            "2-M16_20260317_190134": 1644,
        },
        "channel_names": ["TP9", "AF7", "AF8", "TP10"],
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise SystemExit(f"Unexpected {key}: {summary.get(key)} != {expected}")

    if args.check_only:
        errors = _compare(payload, out_path)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(f"OK: {out_path.relative_to(REPO_ROOT)} matches the rebuilt 2-M16 reference dataset")
        print(f"sha256: {_sha256_file(out_path)}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote: {out_path.relative_to(REPO_ROOT)}")
    print(f"sha256: {_sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
