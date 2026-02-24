from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from muse_streaming.io_paths import SessionDirPaths, prepare_session_dir_paths


def _raw_dtype(channel_count: int) -> np.dtype:
    return np.dtype(
        [
            ("seq", "<i8"),
            ("lsl_ts_raw", "<f8"),
            ("lsl_ts_mono", "<f8"),
            ("local_ts", "<f8"),
            ("flags", "<i8"),
            ("segment_id", "<i8"),
            ("clamped", "<i1"),
            ("sample", "<f8", (channel_count,)),
        ]
    )


class RawShardWriter:
    def __init__(
        self,
        *,
        raw_dir: Path,
        channel_count: int,
        shard_size_samples: int,
    ) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.channel_count = int(channel_count)
        self.shard_size_samples = int(shard_size_samples)
        self._dtype = _raw_dtype(self.channel_count)
        self._buffer: list[np.ndarray] = []
        self._buffer_count = 0
        self._shard_index = 0
        self._shard_meta: list[dict[str, object]] = []
        self._last_flush_mono: Optional[float] = None

    @property
    def shard_meta(self) -> list[dict[str, object]]:
        return list(self._shard_meta)

    @property
    def shard_count(self) -> int:
        return int(len(self._shard_meta))

    @property
    def last_flush_mono(self) -> Optional[float]:
        return self._last_flush_mono

    def append(self, records: np.ndarray) -> None:
        if records.size == 0:
            return
        self._buffer.append(records)
        self._buffer_count += int(records.shape[0])
        while self._buffer_count >= self.shard_size_samples:
            self._flush(self.shard_size_samples)

    def flush(self) -> None:
        if self._buffer_count:
            self._flush(self._buffer_count)

    def _flush(self, target_count: int) -> None:
        if not self._buffer_count:
            return
        pieces = []
        remaining = target_count
        while remaining > 0 and self._buffer:
            chunk = self._buffer[0]
            if chunk.shape[0] <= remaining:
                pieces.append(chunk)
                remaining -= chunk.shape[0]
                self._buffer.pop(0)
            else:
                pieces.append(chunk[:remaining])
                self._buffer[0] = chunk[remaining:]
                remaining = 0
        if not pieces:
            return
        records = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        if not records.size:
            return
        self._buffer_count -= int(records.shape[0])
        shard_name = f"eeg_raw_shard_{self._shard_index:03d}.npy"
        final_path = self.raw_dir / shard_name
        tmp_path = self.raw_dir / f"{shard_name}.tmp.npy"
        np.save(tmp_path, records)
        tmp_path.replace(final_path)
        try:
            rel_path = str(final_path.relative_to(self.raw_dir.parent))
        except Exception:
            rel_path = str(final_path)
        self._shard_meta.append(
            {
                "path": rel_path,
                "seq_min": int(records["seq"][0]),
                "seq_max": int(records["seq"][-1]),
                "count": int(records.shape[0]),
                "lsl_ts_mono_start": float(records["lsl_ts_mono"][0]),
                "lsl_ts_mono_end": float(records["lsl_ts_mono"][-1]),
                "local_ts_start": float(records["local_ts"][0]),
                "local_ts_end": float(records["local_ts"][-1]),
            }
        )
        self._last_flush_mono = time.monotonic()
        self._shard_index += 1

    def empty_record_array(self, count: int) -> np.ndarray:
        return np.zeros((count,), dtype=self._dtype)


class SessionWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        subject_id: str,
        session_id: str,
        channel_labels: Iterable[str],
        sampling_rate: float,
        timebase_version: str,
        shard_size_samples: int = 2048,
        resume: bool = False,
        mode: Optional[str] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._paths, self.resumed, self.reason = prepare_session_dir_paths(
            output_root=output_root,
            subject_id=subject_id,
            session_id=session_id,
            resume=resume,
        )
        self.subject_id = subject_id
        self.session_id = self._paths.session_id
        self.channel_labels = list(channel_labels)
        self.sampling_rate = float(sampling_rate)
        self.timebase_version = timebase_version
        self.mode = mode
        self._shard_writer = RawShardWriter(
            raw_dir=self._paths.raw_dir,
            channel_count=len(self.channel_labels),
            shard_size_samples=shard_size_samples,
        )
        self._seq_min: Optional[int] = None
        self._seq_max: Optional[int] = None
        self._missing_seq_count = 0
        self._out_of_order_count = 0
        self._total_samples = 0
        self._last_seq: Optional[int] = None
        self._events_handle = None
        self._timebase_ranges: list[dict[str, object]] = []
        self._meta: dict[str, object] = {}
        self._write_meta()
        self._ensure_manifest_stub()

    @property
    def paths(self) -> SessionDirPaths:
        return self._paths

    def _write_meta(self) -> None:
        self._paths.session_dir.mkdir(parents=True, exist_ok=True)
        self._paths.raw_dir.mkdir(parents=True, exist_ok=True)
        self._paths.events_dir.mkdir(parents=True, exist_ok=True)
        events_path = self._paths.events_dir / "events.jsonl"
        if not events_path.exists():
            events_path.write_text("")
        meta = {
            "schema_version": 1,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sampling_rate": self.sampling_rate,
            "channel_labels": self.channel_labels,
            "timebase_version": self.timebase_version,
            "mode": self.mode,
            "complete": False,
        }
        meta.update(self._meta)
        self._paths.meta_path.write_text(json.dumps(meta, indent=2))

    def update_meta(self, extra: dict[str, object]) -> None:
        if not extra:
            return
        with self._lock:
            self._meta.update(extra)
            self._write_meta()

    def update_manifest_files(self, extra_files: dict[str, str]) -> None:
        """
        Update/merge the manifest.json 'files' mapping without finalizing the session.

        This lets Step 1 declare auxiliary artifacts (e.g., inspection CSVs) immediately,
        so the manifest remains a reliable index even for short/aborted sessions.
        """
        if not extra_files:
            return
        with self._lock:
            self._ensure_manifest_stub()
            try:
                manifest = json.loads(self._paths.manifest_path.read_text())
            except Exception:
                manifest = {}
            files = manifest.get("files")
            if not isinstance(files, dict):
                files = {}
            for k, v in extra_files.items():
                if not k:
                    continue
                files[str(k)] = str(v)
            manifest["files"] = files
            _write_json_atomic(self._paths.manifest_path, manifest)

    def _ensure_manifest_stub(self) -> None:
        if self._paths.manifest_path.exists():
            return
        stub = {
            "schema_version": 1,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "seq_min": None,
            "seq_max": None,
            "expected_sample_count": 0,
            "actual_sample_count": 0,
            "missing_seq_count": 0,
            "out_of_order_count": 0,
            "termination_reason": "in_progress",
            "shard_list": [],
            "files": {
                "meta": "meta.json",
                "events_jsonl": "events/events.jsonl",
                "timebase_report": "timebase_report.json",
            },
        }
        _write_json_atomic(self._paths.manifest_path, stub)

    def append_packets(self, packets: Iterable[object]) -> None:
        if packets is None:
            return
        packets = list(packets)
        if not packets:
            return
        with self._lock:
            record_arr = self._shard_writer.empty_record_array(len(packets))
            for idx, packet in enumerate(packets):
                seq = int(packet.seq)
                record_arr["seq"][idx] = seq
                record_arr["lsl_ts_raw"][idx] = float(packet.lsl_ts_raw)
                record_arr["lsl_ts_mono"][idx] = float(packet.lsl_ts_mono)
                record_arr["local_ts"][idx] = float(packet.local_ts)
                record_arr["flags"][idx] = int(packet.flags)
                record_arr["segment_id"][idx] = int(packet.segment_id)
                record_arr["clamped"][idx] = int(bool(packet.clamped))
                record_arr["sample"][idx] = np.asarray(packet.sample, dtype=float)

                if self._last_seq is not None:
                    if seq <= self._last_seq:
                        self._out_of_order_count += 1
                    else:
                        gap = seq - self._last_seq - 1
                        if gap > 0:
                            self._missing_seq_count += gap
                self._last_seq = seq

            self._seq_min = (
                int(record_arr["seq"][0])
                if self._seq_min is None
                else min(self._seq_min, int(record_arr["seq"][0]))
            )
            self._seq_max = (
                int(record_arr["seq"][-1])
                if self._seq_max is None
                else max(self._seq_max, int(record_arr["seq"][-1]))
            )
            self._total_samples += int(record_arr.shape[0])
            self._shard_writer.append(record_arr)

    def append_event(self, payload: dict[str, object]) -> None:
        with self._lock:
            if self._events_handle is None:
                self._paths.events_dir.mkdir(parents=True, exist_ok=True)
                self._events_handle = (self._paths.events_dir / "events.jsonl").open(
                    "a", encoding="utf-8"
                )
            self._events_handle.write(json.dumps(payload))
            self._events_handle.write("\n")
            self._events_handle.flush()
            try:
                os.fsync(self._events_handle.fileno())
            except Exception:
                pass

    def flush(self) -> None:
        with self._lock:
            self._shard_writer.flush()

    def shard_metrics(self) -> dict[str, Optional[float] | int]:
        with self._lock:
            return {
                "shard_count": int(self._shard_writer.shard_count),
                "last_flush_mono": self._shard_writer.last_flush_mono,
            }

    def finalize(
        self, termination_reason: str = "normal", extra_manifest: Optional[dict[str, object]] = None
    ) -> None:
        with self._lock:
            self._shard_writer.flush()
            if self._events_handle is not None:
                self._events_handle.flush()
                self._events_handle.close()
                self._events_handle = None
            expected = (
                int(self._seq_max - self._seq_min + 1)
                if self._seq_min is not None and self._seq_max is not None
                else 0
            )
            missing_seq_count = max(
                int(self._missing_seq_count), int(max(0, expected - self._total_samples))
            )
            manifest = {
                "schema_version": 1,
                "subject_id": self.subject_id,
                "session_id": self.session_id,
                "seq_min": self._seq_min,
                "seq_max": self._seq_max,
                "expected_sample_count": expected,
                "actual_sample_count": int(self._total_samples),
                "missing_seq_count": missing_seq_count,
                "out_of_order_count": int(self._out_of_order_count),
                "termination_reason": termination_reason,
                "shard_list": self._shard_writer.shard_meta,
                "files": {
                    "meta": "meta.json",
                    "events_jsonl": "events/events.jsonl",
                    "timebase_report": "timebase_report.json",
                },
            }
            if extra_manifest:
                extra = dict(extra_manifest)
                files_extra = extra.pop("files", None)
                if isinstance(files_extra, dict):
                    merged_files = dict(manifest.get("files") or {})
                    merged_files.update(files_extra)
                    manifest["files"] = merged_files
                elif files_extra is not None:
                    manifest["files"] = files_extra
                manifest.update(extra)
            _write_json_atomic(self._paths.manifest_path, manifest)

            self._meta["complete"] = True
            self._meta["sample_count"] = int(self._total_samples)
            self._write_meta()

            timebase_report = {
                "subject_id": self.subject_id,
                "session_id": self.session_id,
                "ranges": self._shard_writer.shard_meta,
            }
            _write_json_atomic(self._paths.timebase_report_path, timebase_report)


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)
