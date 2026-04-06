from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from utils.model_outputs import infer_output_dims_from_state_dict, unpack_model_outputs

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


def _load_runtime_utils() -> tuple[Any, Any, Any]:
    try:
        from utils.runtime_utils import (
            apply_channel_normalizer,
            load_normalizer,
            load_temperature_scaling,
        )
    except Exception as exc:
        message = (
            "Replay visualization dependencies failed to import. "
            "Recreate the Python 3.11 environment with ./scripts/setup_venv.sh. "
            "Original error: "
            f"{exc.__class__.__name__}: {exc}"
        )
        raise RuntimeError(message) from exc
    return apply_channel_normalizer, load_normalizer, load_temperature_scaling


def _ensure_windows_ntc(X: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected X to be 3D, got shape {arr.shape}")
    channel_names = meta.get("channel_names")
    if channel_names is not None:
        try:
            channel_count = int(len(np.asarray(channel_names)))
        except Exception:
            channel_count = None
        else:
            if arr.shape[2] == channel_count:
                return arr
            if arr.shape[1] == channel_count and arr.shape[2] != channel_count:
                return np.transpose(arr, (0, 2, 1))
    if arr.shape[1] <= 16 and arr.shape[2] > 16:
        return np.transpose(arr, (0, 2, 1))
    return arr


def _meta_array(
    meta: dict[str, Any],
    key: str,
    n: int,
    *,
    dtype: Any | None = None,
) -> np.ndarray | None:
    if key not in meta:
        return None
    arr = np.asarray(meta[key])
    if arr.ndim == 0:
        arr = np.full(n, arr.item(), dtype=arr.dtype if dtype is None else dtype)
    elif len(arr) != n:
        return None
    if dtype is None:
        return arr
    return np.asarray(arr, dtype=dtype)


def _load_state_dict(torch: Any, model_path: str | Path) -> dict[str, Any]:
    path = str(model_path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass(frozen=True)
class ReplayCheckpointSpec:
    n_fingers: int
    n_actions: int
    has_applicability_head: bool


def _infer_replay_checkpoint_spec(state_dict: dict[str, Any]) -> ReplayCheckpointSpec:
    n_fingers, n_actions, has_applicability_head = infer_output_dims_from_state_dict(
        state_dict
    )
    return ReplayCheckpointSpec(
        n_fingers=int(n_fingers),
        n_actions=int(n_actions),
        has_applicability_head=bool(has_applicability_head),
    )


@dataclass
class ReplayVisualizer:
    npz_path: str
    model_path: str
    scaler_path: str

    def __post_init__(self) -> None:
        torch = _load_torch()
        (
            apply_channel_normalizer,
            load_normalizer,
            load_temperature_scaling,
        ) = _load_runtime_utils()
        from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet

        self._torch = torch
        self._apply_channel_normalizer = apply_channel_normalizer
        npz = np.load(self.npz_path, allow_pickle=True)
        if "X" not in npz:
            raise ValueError("NPZ file missing 'X' windows array.")
        self.meta = {key: npz[key] for key in npz.files if key != "X"}
        self.X = _ensure_windows_ntc(npz["X"], self.meta)
        n = int(self.X.shape[0])
        self.y_action = _meta_array(self.meta, "y_action", n, dtype=np.int64)
        self.y_finger = _meta_array(self.meta, "y_finger", n, dtype=np.int64)
        self.window_start = _meta_array(self.meta, "window_start", n, dtype=np.float32)
        if self.window_start is None:
            self.window_start = _meta_array(
                self.meta, "window_start_s", n, dtype=np.float32
            )
        self.window_end = _meta_array(self.meta, "window_end", n, dtype=np.float32)
        if self.window_end is None:
            self.window_end = _meta_array(
                self.meta, "window_end_s", n, dtype=np.float32
            )
        self.trial_ids = _meta_array(self.meta, "trial_id", n, dtype=np.int64)
        if self.trial_ids is None:
            self.trial_ids = _meta_array(self.meta, "trial_ids", n, dtype=np.int64)
        self.session_ids = _meta_array(self.meta, "session_id", n)
        if self.session_ids is None:
            self.session_ids = _meta_array(self.meta, "session_ids", n)
        if self.session_ids is not None:
            self.session_ids = self.session_ids.astype("U")
        self.event_ids = _meta_array(self.meta, "event_id", n, dtype=np.int64)
        if self.event_ids is None:
            self.event_ids = _meta_array(self.meta, "event_ids", n, dtype=np.int64)
        self.event_onset_s = _meta_array(
            self.meta, "event_onset_s", n, dtype=np.float32
        )
        if self.window_start is not None:
            order = np.argsort(self.window_start.astype(np.float64), kind="stable")
            self.X = self.X[order]
            self.window_start = self.window_start[order]
            if self.window_end is not None:
                self.window_end = self.window_end[order]
            if self.y_action is not None:
                self.y_action = self.y_action[order]
            if self.y_finger is not None:
                self.y_finger = self.y_finger[order]
            if self.trial_ids is not None:
                self.trial_ids = self.trial_ids[order]
            if self.session_ids is not None:
                self.session_ids = self.session_ids[order]
            if self.event_ids is not None:
                self.event_ids = self.event_ids[order]
            if self.event_onset_s is not None:
                self.event_onset_s = self.event_onset_s[order]
        self.window_count = int(self.X.shape[0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = _load_state_dict(torch, self.model_path)
        checkpoint_spec = _infer_replay_checkpoint_spec(state_dict)
        self.model = CNNLSTMFingerActionNet(
            n_channels=self.X.shape[2],
            n_fingers=checkpoint_spec.n_fingers,
            n_actions=checkpoint_spec.n_actions,
            finger_applicability_head=checkpoint_spec.has_applicability_head,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.scaler = load_normalizer(Path(self.scaler_path))
        temperature_path = Path(self.model_path).expanduser().resolve().parent / "temperature_scaling.json"
        self.temperature_state = (
            load_temperature_scaling(temperature_path)
            if temperature_path.exists()
            else None
        )

    def _get_window(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.window_count:
            raise IndexError("Window index out of range.")
        window = np.asarray(self.X[idx], dtype=np.float32)
        if window.ndim != 2:
            raise ValueError("Expected window shape [T, C].")
        return window

    def _standardize(self, window: np.ndarray) -> np.ndarray:
        return self._apply_channel_normalizer(window, self.scaler)

    def _tensor_from_window(self, idx: int) -> Any:
        window = self._standardize(self._get_window(idx))
        torch = self._torch
        x = torch.tensor(window, dtype=torch.float32, device=self.device)
        return x.unsqueeze(0)

    def ground_truth_pair(self, idx: int) -> tuple[Optional[int], Optional[int]]:
        if idx < 0 or idx >= self.window_count:
            raise IndexError("Window index out of range.")
        truth_action_id = (
            int(self.y_action[idx]) if self.y_action is not None else None
        )
        truth_finger_id = (
            int(self.y_finger[idx]) if self.y_finger is not None else None
        )
        return truth_action_id, truth_finger_id

    def replay_runtime_inputs(self) -> dict[str, Any]:
        if self.window_start is None or self.window_end is None:
            raise ValueError(
                "Replay NPZ is missing window_start/window_end metadata required for "
                "Step 7-equivalent replay actuation."
            )
        if self.y_action is None or self.y_finger is None:
            raise ValueError(
                "Replay NPZ is missing y_action/y_finger labels required for replay "
                "correctness and actuation preview."
            )
        return {
            "X": self.X,
            "window_start_s": self.window_start,
            "window_end_s": self.window_end,
            "y_action_true": self.y_action,
            "y_finger_true": self.y_finger,
            "trial_ids": self.trial_ids,
            "session_ids": self.session_ids,
            "event_ids": self.event_ids,
            "event_onset_s": self.event_onset_s,
            "scaler": self.scaler,
            "model": self.model,
            "device": self.device,
            "temperature_state": self.temperature_state,
        }

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
        _, action_logits, _ = unpack_model_outputs(self.model(x))
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
