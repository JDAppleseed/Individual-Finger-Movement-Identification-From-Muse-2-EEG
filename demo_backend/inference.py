from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from demo_backend.utils_demo import apply_channel_normalizer, compute_health_score


@dataclass
class InferenceConfig:
    base_threshold: float = 0.75
    uncertainty_weight: float = 0.5
    stability_frames: int = 3
    mc_passes: int = 10


def _mc_predict(model, x_BTC: torch.Tensor, passes: int) -> Dict[str, torch.Tensor]:
    if hasattr(model, "mc_forward"):
        with torch.inference_mode():
            return model.mc_forward(x_BTC, passes=passes)

    was_training = model.training
    model.train()
    finger_probs = []
    action_probs = []
    with torch.inference_mode():
        for _ in range(passes):
            finger_logits, action_logits = model(x_BTC)
            finger_probs.append(torch.softmax(finger_logits, dim=1))
            action_probs.append(torch.softmax(action_logits, dim=1))
    if not was_training:
        model.eval()

    finger_probs = torch.stack(finger_probs, dim=0)
    action_probs = torch.stack(action_probs, dim=0)

    finger_mean = finger_probs.mean(dim=0)
    action_mean = action_probs.mean(dim=0)
    finger_std = finger_probs.std(dim=0)
    action_std = action_probs.std(dim=0)

    return {
        "finger_mean": finger_mean,
        "action_mean": action_mean,
        "finger_std": finger_std,
        "action_std": action_std,
    }


class InferenceEngine:
    def __init__(
        self,
        model: Optional[torch.nn.Module],
        normalizer: Any,
        device: torch.device,
        action_names: Dict[int, str],
        finger_names: Dict[int, str],
        config: Optional[InferenceConfig] = None,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.action_names = action_names
        self.finger_names = finger_names
        self.config = config or InferenceConfig()
        self._stability = deque(maxlen=self.config.stability_frames)
        self._input_np: Optional[np.ndarray] = None
        self._input_tensor: Optional[torch.Tensor] = None
        self._compiled = False

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def set_device(self, device: torch.device) -> None:
        self.device = device
        if self.model is not None:
            self.model.to(self.device)
        self._input_tensor = None

    def compile_model(self) -> bool:
        if self.model is None or self._compiled:
            return False
        if not hasattr(torch, "compile"):
            return False
        try:
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]
            self._compiled = True
            return True
        except Exception:
            return False

    def _ensure_buffers(self, shape: Tuple[int, int]) -> None:
        if self._input_np is None or self._input_np.shape != shape:
            self._input_np = np.empty(shape, dtype=np.float32)
        expected = (1,) + shape
        if (
            self._input_tensor is None
            or tuple(self._input_tensor.shape) != expected
            or self._input_tensor.device != self.device
        ):
            self._input_tensor = torch.empty(
                expected, dtype=torch.float32, device=self.device
            )

    def _normalize_window(self, window_TxC: np.ndarray) -> np.ndarray:
        window = np.asarray(window_TxC, dtype=np.float32)
        self._ensure_buffers(window.shape)
        if self._input_np is None:
            return window
        np.copyto(self._input_np, window)
        if self.normalizer is None:
            return self._input_np
        if (
            isinstance(self.normalizer, dict)
            and "mean" in self.normalizer
            and "std" in self.normalizer
        ):
            mean = np.asarray(self.normalizer["mean"], dtype=np.float32)
            std = np.asarray(self.normalizer["std"], dtype=np.float32)
            std = np.where(std == 0, 1.0, std)
            self._input_np -= mean
            self._input_np /= std
            return self._input_np
        if hasattr(self.normalizer, "mean_") and hasattr(self.normalizer, "scale_"):
            mean = np.asarray(self.normalizer.mean_, dtype=np.float32)
            scale = np.asarray(self.normalizer.scale_, dtype=np.float32)
            scale = np.where(scale == 0, 1.0, scale)
            self._input_np -= mean
            self._input_np /= scale
            return self._input_np
        return apply_channel_normalizer(window, self.normalizer)

    def _to_tensor(self, window_TxC: np.ndarray) -> torch.Tensor:
        if self._input_np is not None and window_TxC is self._input_np:
            assert self._input_tensor is not None
            self._input_tensor[0].copy_(torch.from_numpy(window_TxC))
            return self._input_tensor
        return torch.tensor(
            window_TxC, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def predict_proba(
        self, window_TxC: np.ndarray
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], float, float, Dict[str, Any]
    ]:
        if self.model is None:
            return (
                None,
                None,
                1.0,
                1.0,
                {"health_score": compute_health_score(window_TxC)},
            )

        normalized = self._normalize_window(window_TxC)
        x = self._to_tensor(normalized)

        passes = int(self.config.mc_passes)
        if passes <= 1:
            was_training = self.model.training
            self.model.eval()
            with torch.inference_mode():
                finger_logits, action_logits = self.model(x)
                finger_probs = torch.softmax(finger_logits, dim=1)
                action_probs = torch.softmax(action_logits, dim=1)
            if was_training:
                self.model.train()
            action_mean = action_probs.squeeze(0).detach().cpu().numpy()
            finger_mean = finger_probs.squeeze(0).detach().cpu().numpy()
            action_std = np.zeros_like(action_mean)
            finger_std = np.zeros_like(finger_mean)
        else:
            mc = _mc_predict(self.model, x, passes=passes)
            action_mean = mc["action_mean"].squeeze(0).detach().cpu().numpy()
            finger_mean = mc["finger_mean"].squeeze(0).detach().cpu().numpy()
            action_std = mc["action_std"].squeeze(0).detach().cpu().numpy()
            finger_std = mc["finger_std"].squeeze(0).detach().cpu().numpy()

        action_uncertainty = float(np.mean(action_std))
        finger_uncertainty = float(np.mean(finger_std))

        diagnostics = {
            "health_score": compute_health_score(window_TxC),
        }
        return (
            action_mean,
            finger_mean,
            action_uncertainty,
            finger_uncertainty,
            diagnostics,
        )

    def predict(
        self, window_TxC: np.ndarray
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        if self.model is None:
            return self._empty_prediction(window_TxC)

        (
            action_mean,
            finger_mean,
            action_uncertainty,
            finger_uncertainty,
            diagnostics,
        ) = self.predict_proba(window_TxC)
        if action_mean is None or finger_mean is None:
            return self._empty_prediction(window_TxC)

        action_id = int(np.argmax(action_mean))
        finger_id = int(np.argmax(finger_mean))

        action_confidence = float(action_mean[action_id])
        finger_confidence = float(finger_mean[finger_id])

        adaptive = min(
            0.99,
            max(
                self.config.base_threshold,
                self.config.base_threshold
                + self.config.uncertainty_weight * action_uncertainty,
            ),
        )

        self._stability.append(action_id)
        stability_ok = (
            len(self._stability) == self.config.stability_frames
            and len(set(self._stability)) == 1
        )

        velocity = 0.0
        if action_id != 0:
            velocity = action_confidence * (1.0 - action_uncertainty)

        prediction = {
            "action_id": action_id,
            "action_name": self.action_names.get(action_id, "UNKNOWN"),
            "finger_id": finger_id,
            "finger_name": self.finger_names.get(finger_id, "UNKNOWN"),
            "action_confidence": action_confidence,
            "action_uncertainty": action_uncertainty,
            "finger_confidence": finger_confidence,
            "finger_uncertainty": finger_uncertainty,
        }

        safety = {
            "base_threshold": self.config.base_threshold,
            "adaptive_threshold": adaptive,
            "allow_actuation": action_confidence >= adaptive and stability_ok,
            "stability_frames": self.config.stability_frames,
            "stability_ok": stability_ok,
            "velocity": velocity,
        }

        return prediction, safety, diagnostics

    def _empty_prediction(self, window_TxC: np.ndarray):
        prediction = {
            "action_id": -1,
            "action_name": "UNAVAILABLE",
            "finger_id": -1,
            "finger_name": "UNAVAILABLE",
            "action_confidence": 0.0,
            "action_uncertainty": 1.0,
            "finger_confidence": 0.0,
            "finger_uncertainty": 1.0,
        }
        safety = {
            "base_threshold": self.config.base_threshold,
            "adaptive_threshold": self.config.base_threshold,
            "allow_actuation": False,
            "stability_frames": self.config.stability_frames,
            "stability_ok": False,
            "velocity": 0.0,
        }
        diagnostics = {
            "health_score": compute_health_score(window_TxC),
        }
        return prediction, safety, diagnostics
