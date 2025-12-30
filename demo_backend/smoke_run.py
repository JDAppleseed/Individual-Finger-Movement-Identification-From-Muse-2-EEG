from pathlib import Path

import torch

from demo_backend.inference import InferenceConfig, InferenceEngine
from demo_backend.replay import ReplaySource
from demo_backend.utils_demo import ensure_repo_on_path, load_normalizer, resolve_device

ensure_repo_on_path()

from utils.label_schema import ACTION_NAMES, FINGER_NAMES  # noqa: E402
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    npz_path = root / "eeg_windows.npz"
    model_path = root / "finger_action_model.pt"

    if not npz_path.exists():
        raise SystemExit("eeg_windows.npz not found")
    if not model_path.exists():
        raise SystemExit("finger_action_model.pt not found")

    normalizer = load_normalizer(root / "scaler.save")
    state = torch.load(model_path, map_location="cpu")
    n_fingers = int(state["finger_head.weight"].shape[0])
    n_actions = int(state["action_head.weight"].shape[0])
    model = CNNLSTMFingerActionNet(n_channels=4, n_fingers=n_fingers, n_actions=n_actions)
    model.load_state_dict(state)

    engine = InferenceEngine(
        model=model,
        normalizer=normalizer,
        device=resolve_device("cpu"),
        action_names=ACTION_NAMES,
        finger_names=FINGER_NAMES,
        config=InferenceConfig(mc_passes=3),
    )

    replay = ReplaySource(npz_path)
    for idx, (window, meta) in enumerate(replay.iter_windows()):
        pred, safety, diag = engine.predict(window)
        print(f"tick {idx} | action={pred['action_name']} conf={pred['action_confidence']:.2f} health={diag['health_score']:.2f}")
        if idx >= 4:
            break


if __name__ == "__main__":
    main()
