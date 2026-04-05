from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Optional

import numpy as np

from utils.runtime_utils import now_utc_iso


def sha256_file(path: Path | str | None) -> Optional[str]:
    if path is None:
        return None
    try:
        resolved = Path(path).expanduser().resolve()
    except Exception:
        resolved = Path(path).expanduser()
    if not resolved.exists() or not resolved.is_file():
        return None
    hasher = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_required_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        labels = [str(item).strip() for item in value]
    else:
        labels = [part.strip() for part in str(value).split(",")]
    return [label for label in labels if label]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def write_json(path: Path, payload: Any) -> None:
    path = Path(path).expanduser().resolve()
    _write_text_atomic(
        path,
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
    )


def write_jsonl_row(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text())


def load_capture_records(capture_dir: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capture_root = Path(capture_dir).expanduser().resolve()
    manifest_path = capture_root / "capture_manifest.json"
    records_path = capture_root / "captured_windows.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    payload = load_json(records_path) if records_path.exists() else {}
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return manifest if isinstance(manifest, dict) else {}, list(records)


@dataclass(frozen=True)
class ParityCaptureSettings:
    enabled: bool = False
    max_windows: int = 64
    flush_every: int = 8


class LiveParityCapture:
    def __init__(
        self,
        *,
        root_dir: Path,
        settings: ParityCaptureSettings,
        manifest_seed: Optional[dict[str, Any]] = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.settings = ParityCaptureSettings(
            enabled=bool(settings.enabled),
            max_windows=max(1, int(settings.max_windows)),
            flush_every=max(1, int(settings.flush_every)),
        )
        self.records: list[dict[str, Any]] = []
        self.total_seen = 0
        self.flush_count = 0
        self.created_at = now_utc_iso()
        self.manifest_seed = dict(manifest_seed or {})
        self.capture_dir = self.root_dir / "parity_capture"
        self.manifest_path = self.capture_dir / "capture_manifest.json"
        self.records_path = self.capture_dir / "captured_windows.json"
        if self.settings.enabled:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            self._persist(increment_flush=False)

    def add(self, payload: dict[str, Any]) -> None:
        if not self.settings.enabled:
            return
        self.total_seen += 1
        self.records.append(json_ready(payload))
        if len(self.records) > self.settings.max_windows:
            self.records = self.records[-self.settings.max_windows :]
        # Persist every accepted window atomically so interruption cannot erase
        # the only parity evidence from a live run.
        self._persist(increment_flush=((self.total_seen % self.settings.flush_every) == 0))

    def flush(self) -> None:
        if not self.settings.enabled:
            return
        self._persist(increment_flush=True)

    def _persist(self, *, increment_flush: bool) -> None:
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        if increment_flush:
            self.flush_count += 1
        write_json(self.records_path, {"records": self.records})
        manifest = {
            "created_at": self.created_at,
            "updated_at": now_utc_iso(),
            "settings": asdict(self.settings),
            "record_count": int(len(self.records)),
            "total_seen": int(self.total_seen),
            "flush_count": int(self.flush_count),
            "records_path": str(self.records_path),
            "records_sha256": sha256_file(self.records_path),
            "records_candidate_indices": [
                int(row.get("candidate_index", -1))
                for row in self.records
                if row.get("candidate_index") is not None
            ],
            "manifest_seed": json_ready(self.manifest_seed),
        }
        write_json(self.manifest_path, manifest)

    def close(self) -> None:
        self.flush()


def summarize_counter_rows(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key) or "none")
        counts[label] = int(counts.get(label, 0) + 1)
    return dict(sorted(counts.items(), key=lambda item: item[0]))
