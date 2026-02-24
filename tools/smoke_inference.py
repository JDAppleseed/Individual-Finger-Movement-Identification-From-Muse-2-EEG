#!/usr/bin/env python
"""
Smoke inference: load a trained model, one window, run preprocessing + forward + postprocess.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from utils.runtime_utils import apply_channel_normalizer, load_normalizer


def main():
    from utils.postprocess import (
        PostprocessSettings,
        PostprocessState,
        postprocess_predictions,
    )
    from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
    from utils.label_schema import ACTION_NAMES, FINGER_NAMES
    from utils.sequence_data import load_sequence_npz

    parser = argparse.ArgumentParser(description="Smoke inference on a single window")
    parser.add_argument(
        "--npz", type=str, default="eeg_windows.npz", help="Path to window dataset"
    )
    parser.add_argument(
        "--model", type=str, default="finger_action_model.pt", help="Model weights path"
    )
    parser.add_argument("--scaler", type=str, default="scaler.npz", help="Scaler path")
    parser.add_argument("--index", type=int, default=0, help="Window index to use")
    parser.add_argument(
        "--n-fingers", type=int, default=6, help="Number of finger classes"
    )
    parser.add_argument(
        "--n-actions", type=int, default=3, help="Number of action classes"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)

    if not npz_path.exists():
        print(f"ERROR: NPZ not found: {npz_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    X, _, _, _ = load_sequence_npz(str(npz_path))
    if len(X) == 0:
        print("ERROR: No windows found in dataset")
        sys.exit(1)

    idx = int(args.index)
    if idx < 0 or idx >= len(X):
        idx = 0

    window = X[idx]

    scaler = load_normalizer(scaler_path)
    window = apply_channel_normalizer(window, scaler)

    device = torch.device(args.device)
    model = CNNLSTMFingerActionNet(
        n_channels=window.shape[1],
        n_fingers=args.n_fingers,
        n_actions=args.n_actions,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        finger_logits, action_logits = model(xb)
        action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
        finger_probs = torch.softmax(finger_logits, dim=1).cpu().numpy()[0]

    settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        adjacency_enabled=False,
    )
    state = PostprocessState()
    post = postprocess_predictions(action_probs, finger_probs, settings, state)

    raw_action = int(np.argmax(action_probs)) if action_probs.size else 0
    raw_finger = int(np.argmax(finger_probs)) if finger_probs.size else 0
    committed_action = int(post.get("committed_action_id", raw_action))
    committed_finger = int(post.get("committed_finger_id", raw_finger))

    action_label = ACTION_NAMES.get(committed_action, str(committed_action))
    finger_label = FINGER_NAMES.get(committed_finger, str(committed_finger))

    print(
        "Smoke inference OK: "
        f"action={action_label}({committed_action}), "
        f"finger={finger_label}({committed_finger})"
    )


if __name__ == "__main__":
    main()
