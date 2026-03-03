from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from utils.runtime_utils import apply_channel_normalizer, load_normalizer

if TYPE_CHECKING:
    import torch


def _load_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        message = (
            "Torch is required for replay visualization but failed to import. "
            "Use Python 3.11 or 3.12 and run ./scripts/setup_venv.sh to recreate "
            "the virtual environment. Original error: "
            f"{exc.__class__.__name__}: {exc}"
        )
        raise RuntimeError(message) from exc
    return torch


@dataclass
class ReplayVisualizer:
    npz_path: str
    model_path: str
    scaler_path: str

    def __post_init__(self) -> None:
        torch = _load_torch()
        from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet

        self._torch = torch
        npz = np.load(self.npz_path, allow_pickle=True)
        if "X" not in npz:
            raise ValueError("NPZ file missing 'X' windows array.")
        self.X = npz["X"]
        self.window_count = int(self.X.shape[0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNLSTMFingerActionNet(
            n_channels=self.X.shape[2],
            n_fingers=6,
            n_actions=3,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.scaler = load_normalizer(Path(self.scaler_path))

    def _get_window(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.window_count:
            raise IndexError("Window index out of range.")
        window = np.asarray(self.X[idx], dtype=np.float32)
        if window.ndim != 2:
            raise ValueError("Expected window shape [T, C].")
        return window

    def _standardize(self, window: np.ndarray) -> np.ndarray:
        return apply_channel_normalizer(window, self.scaler)

    def _tensor_from_window(self, idx: int) -> Any:
        window = self._standardize(self._get_window(idx))
        torch = self._torch
        x = torch.tensor(window, dtype=torch.float32, device=self.device)
        return x.unsqueeze(0)

    def feature_map(self, idx: int, layer_idx: int) -> Optional[np.ndarray]:
        torch = self._torch
        x = self._tensor_from_window(idx)
        x = x.permute(0, 2, 1)
        conv_outputs = []
        for layer in self.model.conv:
            x = layer(x)
            if isinstance(layer, torch.nn.Conv1d):
                conv_outputs.append(x.detach().cpu())
        if not conv_outputs:
            return None
        if layer_idx < 0 or layer_idx >= len(conv_outputs):
            layer_idx = len(conv_outputs) - 1
        feature = conv_outputs[layer_idx].squeeze(0).numpy()
        return feature

    def hidden_magnitude(self, idx: int) -> Optional[np.ndarray]:
        torch = self._torch
        x = self._tensor_from_window(idx)
        x = x.permute(0, 2, 1)
        x = self.model.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.model.lstm(x)
        hidden_mag = torch.linalg.norm(out, dim=2).squeeze(0)
        return hidden_mag.detach().cpu().numpy()

    def saliency(self, idx: int) -> Optional[np.ndarray]:
        torch = self._torch
        x = self._tensor_from_window(idx)
        x.requires_grad = True
        finger_logits, action_logits = self.model(x)
        target_idx = int(torch.argmax(action_logits, dim=1).item())
        loss = action_logits[0, target_idx]
        loss.backward()
        grad = x.grad.detach().cpu().numpy()[0]
        return np.abs(grad)

    def prediction_timeline(self, idx: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Returns per-timestep (finger_probs, action_probs) for the window.
        Shapes: [T, n_fingers], [T, n_actions]
        """
        torch = self._torch
        with torch.no_grad():
            x = self._tensor_from_window(idx)
            x = x.permute(0, 2, 1)
            x = self.model.conv(x)
            x = x.permute(0, 2, 1)
            out, _ = self.model.lstm(x)
            out = self.model.head_dropout(out)
            finger_logits = self.model.finger_head(out)
            action_logits = self.model.action_head(out)
            finger_probs = torch.softmax(finger_logits, dim=2).squeeze(0).cpu().numpy()
            action_probs = torch.softmax(action_logits, dim=2).squeeze(0).cpu().numpy()
        return finger_probs, action_probs
