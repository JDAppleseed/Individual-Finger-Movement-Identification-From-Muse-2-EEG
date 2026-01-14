"""Export NN Visualizer timeline weights from checkpoint files."""

from __future__ import annotations

import argparse
import json

import torch

from demo_backend.nn_vis.extract import extract_weights
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from demo_backend.utils_demo import repo_root


def _load_model_from_state(state_dict: dict) -> CNNLSTMFingerActionNet:
    n_fingers = int(state_dict["finger_head.weight"].shape[0])
    n_actions = int(state_dict["action_head.weight"].shape[0])
    model = CNNLSTMFingerActionNet(
        n_channels=4, n_fingers=n_fingers, n_actions=n_actions
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export nnvis timeline snapshots from checkpoints"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="checkpoints",
        help="Checkpoint directory to scan",
    )
    parser.add_argument(
        "--pattern", type=str, default="*.pt", help="Filename glob for checkpoints"
    )
    parser.add_argument(
        "--output", type=str, default="exports/nnvis_timeline", help="Output directory"
    )
    parser.add_argument(
        "--topk", type=int, default=150, help="Top-K edges for LSTM matrices"
    )
    args = parser.parse_args()

    root = repo_root()
    ckpt_dir = (root / args.checkpoints).resolve()
    out_dir = (root / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir}")
        print("Provide a valid --checkpoints path or copy checkpoints into the repo.")
        return

    checkpoints = sorted(ckpt_dir.glob(args.pattern))
    if not checkpoints:
        print(f"No checkpoint files found in {ckpt_dir} with pattern {args.pattern}")
        print("Expected files like epoch_001.pt containing model state_dict.")
        return

    manifest_steps = []
    for idx, ckpt in enumerate(checkpoints):
        state = torch.load(ckpt, map_location="cpu")
        model = _load_model_from_state(state)
        weights = extract_weights(model, quantize=True, downsample=1, topk=args.topk)
        weights_path = out_dir / f"weights_{idx:03d}.npz"
        weights_payload = json.dumps(weights)
        import numpy as np

        np.savez_compressed(weights_path, weights=weights_payload)
        manifest_steps.append(
            {
                "step": idx,
                "label": ckpt.stem,
                "file": weights_path.name,
            }
        )

    manifest = {"steps": manifest_steps}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(manifest_steps)} timeline steps to {out_dir}")


if __name__ == "__main__":
    main()
