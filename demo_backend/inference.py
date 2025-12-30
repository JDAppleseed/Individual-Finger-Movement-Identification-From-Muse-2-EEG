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
        return model.mc_forward(x_BTC, passes=passes)

    was_training = model.training
    model.train()
    finger_probs = []
    action_probs = []
    with torch.no_grad():
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

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def set_device(self, device: torch.device) -> None:
        self.device = device
        if self.model is not None:
            self.model.to(self.device)

    def predict(self, window_TxC: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        if self.model is None:
            return self._empty_prediction(window_TxC)

        window_TxC = apply_channel_normalizer(window_TxC, self.normalizer)
        x = torch.tensor(window_TxC, dtype=torch.float32, device=self.device).unsqueeze(0)

        mc = _mc_predict(self.model, x, passes=self.config.mc_passes)
        action_mean = mc["action_mean"].squeeze(0).detach().cpu().numpy()
        action_std = mc["action_std"].squeeze(0).detach().cpu().numpy()
        finger_mean = mc["finger_mean"].squeeze(0).detach().cpu().numpy()
        finger_std = mc["finger_std"].squeeze(0).detach().cpu().numpy()

        action_id = int(np.argmax(action_mean))
        finger_id = int(np.argmax(finger_mean))

        action_confidence = float(action_mean[action_id])
        action_uncertainty = float(np.mean(action_std))
        finger_confidence = float(finger_mean[finger_id])
        finger_uncertainty = float(np.mean(finger_std))

        adaptive = min(
            0.99,
            max(self.config.base_threshold, self.config.base_threshold + self.config.uncertainty_weight * action_uncertainty),
        )

        self._stability.append(action_id)
        stability_ok = len(self._stability) == self.config.stability_frames and len(set(self._stability)) == 1

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

        diagnostics = {
            "health_score": compute_health_score(window_TxC),
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
