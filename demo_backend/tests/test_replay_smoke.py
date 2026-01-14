from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_replay_smoke():
    from demo_backend.inference import InferenceConfig, InferenceEngine
    from demo_backend.replay import ReplaySource
    from demo_backend.utils_demo import (
        ensure_repo_on_path,
        load_normalizer,
        resolve_device,
    )
    from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
    from utils.label_schema import ACTION_NAMES, FINGER_NAMES

    ensure_repo_on_path()

    root = Path(__file__).resolve().parents[2]
    npz_path = root / "eeg_windows.npz"
    if not npz_path.exists():
        pytest.skip("eeg_windows.npz not found; skipping replay smoke test")

    model_path = root / "finger_action_model.pt"
    if not model_path.exists():
        pytest.skip("Model weights not found; skipping replay smoke test")

    normalizer = load_normalizer(root / "scaler.save")
    model = CNNLSTMFingerActionNet(
        n_channels=4, n_fingers=len(FINGER_NAMES), n_actions=len(ACTION_NAMES)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    engine = InferenceEngine(
        model=model,
        normalizer=normalizer,
        device=resolve_device("cpu"),
        action_names=ACTION_NAMES,
        finger_names=FINGER_NAMES,
        config=InferenceConfig(mc_passes=3),
    )

    replay = ReplaySource(npz_path)
    count = 0
    for window, _ in replay.iter_windows():
        engine.predict(window)
        count += 1
        if count >= 3:
            break

    assert count == 3
