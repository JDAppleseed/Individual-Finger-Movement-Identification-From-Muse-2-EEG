from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from visualization.muse_topomap import compute_bandpower_windows


def _load_tool_module():
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / "experimental_muse_topomaps.py"
    spec = importlib.util.spec_from_file_location("experimental_muse_topomaps", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load tools/experimental_muse_topomaps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_bandpower_windows_tracks_relative_channel_strength():
    fs = 128.0
    t = np.arange(256, dtype=np.float32) / fs
    amps = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    base = np.stack(
        [amp * np.sin(2.0 * np.pi * 10.0 * t) for amp in amps],
        axis=1,
    )
    X = np.stack([base for _ in range(6)], axis=0)

    bandpower = compute_bandpower_windows(
        X,
        fs=fs,
        band=(8.0, 12.0),
        channel_count=4,
    )
    mean_power = bandpower.mean(axis=0)

    assert bandpower.shape == (6, 4)
    assert np.all(np.diff(mean_power) > 0.0)


def test_experimental_muse_topomaps_writes_split_halves_panel(tmp_path, monkeypatch):
    fs = 128.0
    samples = 256
    t = np.arange(samples, dtype=np.float32) / fs
    channel_names = np.array(["TP9", "AF7", "AF8", "TP10"], dtype="U")

    windows = []
    y_action = []
    y_finger = []
    for action_id, freq, amp_scale in ((0, 9.0, 1.0), (1, 10.0, 1.5), (2, 11.0, 2.0)):
        for rep in range(4):
            amps = amp_scale * np.array(
                [1.0 + 0.1 * rep, 1.4 + 0.1 * rep, 1.8 + 0.1 * rep, 2.2 + 0.1 * rep],
                dtype=np.float32,
            )
            window = np.stack(
                [amp * np.sin(2.0 * np.pi * freq * t) for amp in amps],
                axis=1,
            )
            windows.append(window)
            y_action.append(action_id)
            y_finger.append(0 if action_id == 0 else action_id)

    npz_path = tmp_path / "eeg_windows.npz"
    np.savez_compressed(
        npz_path,
        X=np.stack(windows).astype(np.float32),
        y_action=np.asarray(y_action, dtype=np.int64),
        y_finger=np.asarray(y_finger, dtype=np.int64),
        channel_names=channel_names,
        target_fs=np.array([fs], dtype=np.float32),
    )

    out_path = tmp_path / "topomaps.png"
    module = _load_tool_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experimental_muse_topomaps.py",
            "--npz",
            str(npz_path),
            "--out",
            str(out_path),
            "--group-by",
            "action",
            "--split-halves",
        ],
    )

    assert module.main() == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_experimental_muse_topomaps_resolves_session_defaults_from_config(tmp_path, monkeypatch):
    fs = 128.0
    samples = 256
    t = np.arange(samples, dtype=np.float32) / fs
    channel_names = np.array(["TP9", "AF7", "AF8", "TP10"], dtype="U")

    windows = []
    y_action = []
    y_finger = []
    for action_id, freq, amp_scale in ((0, 9.0, 1.0), (1, 10.0, 1.5), (2, 11.0, 2.0)):
        for rep in range(4):
            amps = amp_scale * np.array(
                [1.0 + 0.1 * rep, 1.4 + 0.1 * rep, 1.8 + 0.1 * rep, 2.2 + 0.1 * rep],
                dtype=np.float32,
            )
            window = np.stack(
                [amp * np.sin(2.0 * np.pi * freq * t) for amp in amps],
                axis=1,
            )
            windows.append(window)
            y_action.append(action_id)
            y_finger.append(0 if action_id == 0 else action_id)

    session_dir = tmp_path / "session_001"
    processed_dir = session_dir / "processed"
    reports_dir = session_dir / "reports"
    processed_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    np.savez_compressed(
        processed_dir / "eeg_windows.npz",
        X=np.stack(windows).astype(np.float32),
        y_action=np.asarray(y_action, dtype=np.int64),
        y_finger=np.asarray(y_finger, dtype=np.int64),
        channel_names=channel_names,
        target_fs=np.array([fs], dtype=np.float32),
    )

    config_path = tmp_path / "topomaps.json"
    config_path.write_text(
        json.dumps(
            {
                "settings": {
                    "session_dir": str(session_dir),
                    "suite": True,
                    "band_low": 8.0,
                    "band_high": 12.0,
                }
            }
        )
    )

    module = _load_tool_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experimental_muse_topomaps.py",
            "--config",
            str(config_path),
        ],
    )

    assert module.main() == 0
    assert (reports_dir / "experimental_muse_action_alpha_rest_delta_topomaps.png").exists()
    assert (reports_dir / "experimental_muse_alpha_summary.md").exists()
    assert (reports_dir / "experimental_muse_alpha_summary.json").exists()
