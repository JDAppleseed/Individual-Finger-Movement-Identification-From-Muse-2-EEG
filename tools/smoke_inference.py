#!/usr/bin/env python
"""
Smoke inference: load a trained model, one window, run preprocessing + forward + postprocess.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from utils.runtime_utils import (
    apply_channel_normalizer,
    apply_temperature_to_logits,
    load_normalizer,
)
from utils.model_outputs import infer_output_dims_from_state_dict, unpack_model_outputs


def main():
    from utils.postprocess import (
        PostprocessSettings,
        PostprocessState,
        postprocess_predictions,
    )
    from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
    from utils.label_schema import (
        ACTION_NAMES,
        ACTIVE_FINGER_IDS,
        FINGER_NAMES,
        decode_prediction_pair,
    )
    from utils.live_infer_common import resolve_temperature_path
    from utils.runtime_utils import load_temperature_scaling
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
        "--n-fingers", type=int, default=None, help="Optional finger head size override"
    )
    parser.add_argument(
        "--n-actions", type=int, default=None, help="Optional action head size override"
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

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    inferred_n_fingers, inferred_n_actions, has_applicability_head = (
        infer_output_dims_from_state_dict(state_dict)
    )
    n_fingers = (
        int(args.n_fingers) if args.n_fingers is not None else int(inferred_n_fingers)
    )
    n_actions = (
        int(args.n_actions) if args.n_actions is not None else int(inferred_n_actions)
    )
    if args.n_fingers is not None and n_fingers != inferred_n_fingers:
        print(
            "ERROR: --n-fingers does not match model weights "
            f"({n_fingers} != {inferred_n_fingers})"
        )
        sys.exit(2)
    if args.n_actions is not None and n_actions != inferred_n_actions:
        print(
            "ERROR: --n-actions does not match model weights "
            f"({n_actions} != {inferred_n_actions})"
        )
        sys.exit(2)

    train_config_path = model_path.parent / "train_config.json"
    active_finger_head = None
    if train_config_path.exists():
        try:
            train_config = json.loads(train_config_path.read_text())
            if isinstance(train_config, dict):
                active_finger_head = train_config.get("active_finger_head")
        except Exception:
            active_finger_head = None
    if n_fingers != len(ACTIVE_FINGER_IDS) or active_finger_head is not True:
        print(
            "ERROR: deployment smoke inference requires an active finger head "
            f"with {len(ACTIVE_FINGER_IDS)} outputs; got n_fingers={n_fingers}, "
            f"active_finger_head={active_finger_head}"
        )
        sys.exit(2)
    finger_applicability_head = None
    if train_config_path.exists():
        try:
            train_config = json.loads(train_config_path.read_text())
            if isinstance(train_config, dict):
                finger_applicability_head = train_config.get(
                    "finger_applicability_head"
                )
        except Exception:
            finger_applicability_head = None
    temperature_state = load_temperature_scaling(resolve_temperature_path(model_path.parent))
    if has_applicability_head is not True or finger_applicability_head is not True:
        print(
            "ERROR: deployment smoke inference requires a finger applicability head "
            f"and matching train_config flag; got model_has_applicability_head={has_applicability_head}, "
            f"finger_applicability_head={finger_applicability_head}"
        )
        sys.exit(2)
    if temperature_state is None or not temperature_state.has_applicability_temperature:
        print(
            "ERROR: deployment smoke inference requires applicability temperature scaling."
        )
        sys.exit(2)

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
        n_fingers=n_fingers,
        n_actions=n_actions,
        finger_applicability_head=bool(has_applicability_head),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        finger_logits, action_logits, applicability_logits = unpack_model_outputs(
            model(xb)
        )
        if temperature_state is not None:
            action_logits = apply_temperature_to_logits(
                action_logits, temperature_state.action_temperature
            )
            finger_logits = apply_temperature_to_logits(
                finger_logits, temperature_state.finger_temperature
            )
            if applicability_logits is not None:
                applicability_logits = apply_temperature_to_logits(
                    applicability_logits,
                    temperature_state.applicability_temperature,
                )
        action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
        finger_probs = torch.softmax(finger_logits, dim=1).cpu().numpy()[0]
        applicability_prob = (
            float(torch.sigmoid(applicability_logits).cpu().numpy().reshape(-1)[0])
            if applicability_logits is not None
            else None
        )

    settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        adjacency_enabled=False,
    )
    state = PostprocessState()
    post = postprocess_predictions(
        action_probs,
        finger_probs,
        settings,
        state,
        finger_applicable_prob=applicability_prob,
    )

    raw_action, raw_finger = decode_prediction_pair(action_probs, finger_probs)
    committed_action = int(post.get("committed_action_id", raw_action))
    committed_finger = int(post.get("committed_finger_id", raw_finger))

    action_label = ACTION_NAMES.get(committed_action, str(committed_action))
    finger_label = FINGER_NAMES.get(committed_finger, str(committed_finger))

    print(
        "Smoke inference OK: "
        f"action={action_label}({committed_action}), "
        f"finger={finger_label}({committed_finger}), "
        f"applicability_prob={applicability_prob:.3f}, "
        f"applicability_gate_ok={bool(post.get('applicability_gate_ok', True))}"
    )


if __name__ == "__main__":
    main()
