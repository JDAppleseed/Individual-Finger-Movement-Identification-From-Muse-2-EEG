from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch

from utils.label_schema import decode_prediction_pair, finger_confidence_for_id
from utils.model_outputs import unpack_model_outputs
from utils.default_recipe import LIVE_INFER_RECIPE_DEFAULTS
from utils.runtime_utils import (
    LogitBiasState,
    TemperatureScalingState,
    apply_channel_normalizer,
    apply_logit_bias,
    apply_temperature_to_logits,
    compute_health_score,
)


@dataclass
class InferenceConfig:
    base_threshold: float = float(LIVE_INFER_RECIPE_DEFAULTS["uncertainty_base_threshold"])
    uncertainty_weight: float = float(LIVE_INFER_RECIPE_DEFAULTS["uncertainty_weight"])
    stability_frames: int = int(LIVE_INFER_RECIPE_DEFAULTS["actuation_stability"])
    mc_passes: int = int(LIVE_INFER_RECIPE_DEFAULTS["mc_passes"])


def _mc_predict(
    model,
    x_BTC: torch.Tensor,
    passes: int,
    *,
    temperature_state: Optional[TemperatureScalingState] = None,
    logit_bias_state: Optional[LogitBiasState] = None,
) -> Dict[str, torch.Tensor]:
    action_temp = (
        float(temperature_state.action_temperature)
        if temperature_state is not None
        else 1.0
    )
    finger_temp = (
        float(temperature_state.finger_temperature)
        if temperature_state is not None
        else 1.0
    )
    applicability_temp = (
        float(temperature_state.applicability_temperature)
        if temperature_state is not None
        else 1.0
    )
    use_native_mc = (
        hasattr(model, "mc_forward")
        and logit_bias_state is None
        and abs(action_temp - 1.0) < 1e-6
        and abs(finger_temp - 1.0) < 1e-6
        and abs(applicability_temp - 1.0) < 1e-6
    )
    if use_native_mc:
        with torch.inference_mode():
            return model.mc_forward(x_BTC, passes=passes)

    was_training = model.training
    model.train()
    batch_size = int(x_BTC.shape[0])
    mc_input = x_BTC.repeat((passes, 1, 1)) if passes > 1 else x_BTC
    with torch.inference_mode():
        finger_logits, action_logits, applicability_logits = unpack_model_outputs(
            model(mc_input)
        )
        finger_logits = apply_temperature_to_logits(finger_logits, finger_temp)
        action_logits = apply_temperature_to_logits(action_logits, action_temp)
        if applicability_logits is not None:
            applicability_logits = apply_temperature_to_logits(
                applicability_logits, applicability_temp
            )
        if logit_bias_state is not None:
            finger_logits = apply_logit_bias(
                finger_logits, logit_bias_state.finger_bias
            )
            action_logits = apply_logit_bias(
                action_logits, logit_bias_state.action_bias
            )
        finger_probs = torch.softmax(finger_logits, dim=1).reshape(
            passes, batch_size, -1
        )
        action_probs = torch.softmax(action_logits, dim=1).reshape(
            passes, batch_size, -1
        )
    if not was_training:
        model.eval()

    finger_mean = finger_probs.mean(dim=0)
    action_mean = action_probs.mean(dim=0)
    if passes > 1:
        finger_std = finger_probs.std(dim=0)
        action_std = action_probs.std(dim=0)
    else:
        finger_std = torch.zeros_like(finger_mean)
        action_std = torch.zeros_like(action_mean)

    result = {
        "finger_mean": finger_mean,
        "action_mean": action_mean,
        "finger_std": finger_std,
        "action_std": action_std,
    }
    if applicability_logits is not None:
        applicability_probs = torch.sigmoid(applicability_logits)
        if applicability_probs.ndim == 1:
            applicability_probs = applicability_probs.reshape(passes, batch_size)
        else:
            applicability_probs = applicability_probs.reshape(
                passes, batch_size, -1
            )
        applicability_mean = applicability_probs.mean(dim=0)
        if passes > 1:
            applicability_std = applicability_probs.std(dim=0)
        else:
            applicability_std = torch.zeros_like(applicability_mean)
        result["applicability_mean"] = applicability_mean
        result["applicability_std"] = applicability_std
    return result


class InferenceEngine:
    def __init__(
        self,
        model: Optional[torch.nn.Module],
        normalizer: Any,
        device: torch.device,
        action_names: Dict[int, str],
        finger_names: Dict[int, str],
        config: Optional[InferenceConfig] = None,
        temperature_state: Optional[TemperatureScalingState] = None,
        logit_bias_state: Optional[LogitBiasState] = None,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.action_names = action_names
        self.finger_names = finger_names
        self.config = config or InferenceConfig()
        self.temperature_state = temperature_state
        self.logit_bias_state = logit_bias_state
        self._stability: Deque[int] = deque(maxlen=self.config.stability_frames)
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
            self.model = torch.compile(self.model)
            self._compiled = True
            return True
        except Exception:
            return False

    def _ensure_buffers(self, shape: Tuple[int, int]) -> None:
        if self._input_np is None or self._input_np.shape != shape:
            self._input_np = np.empty(shape, dtype=np.float32)
        if self.device.type == "cpu":
            self._input_tensor = None
            return
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
        return apply_channel_normalizer(window, self.normalizer, out=self._input_np)

    def _to_tensor(self, window_TxC: np.ndarray) -> torch.Tensor:
        host_tensor = torch.from_numpy(np.asarray(window_TxC, dtype=np.float32))
        if self.device.type == "cpu":
            return host_tensor.unsqueeze(0)
        assert self._input_tensor is not None
        self._input_tensor[0].copy_(host_tensor)
        return self._input_tensor

    def prepare_input(self, window_TxC: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        normalized = self._normalize_window(window_TxC)
        return normalized, self._to_tensor(normalized)

    def forward_probabilities(
        self, x_BTC: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.model is None:
            raise RuntimeError("forward_probabilities requires a loaded model.")
        with torch.inference_mode():
            (
                finger_logits,
                action_logits,
                applicability_logits,
            ) = unpack_model_outputs(self.model(x_BTC))
            finger_logits = apply_temperature_to_logits(
                finger_logits,
                self.temperature_state.finger_temperature
                if self.temperature_state is not None
                else 1.0,
            )
            action_logits = apply_temperature_to_logits(
                action_logits,
                self.temperature_state.action_temperature
                if self.temperature_state is not None
                else 1.0,
            )
            if applicability_logits is not None:
                applicability_logits = apply_temperature_to_logits(
                    applicability_logits,
                    self.temperature_state.applicability_temperature
                    if self.temperature_state is not None
                    else 1.0,
                )
            if self.logit_bias_state is not None:
                finger_logits = apply_logit_bias(
                    finger_logits, self.logit_bias_state.finger_bias
                )
                action_logits = apply_logit_bias(
                    action_logits, self.logit_bias_state.action_bias
                )
            finger_probs = torch.softmax(finger_logits, dim=1)
            action_probs = torch.softmax(action_logits, dim=1)
            applicability_probs = (
                torch.sigmoid(applicability_logits)
                if applicability_logits is not None
                else None
            )
        return finger_probs, action_probs, applicability_probs

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

        _, x = self.prepare_input(window_TxC)

        passes = int(self.config.mc_passes)
        if passes <= 1:
            was_training = self.model.training
            self.model.eval()
            finger_probs, action_probs, applicability_probs = self.forward_probabilities(x)
            if was_training:
                self.model.train()
            action_mean = action_probs.squeeze(0).detach().cpu().numpy()
            finger_mean = finger_probs.squeeze(0).detach().cpu().numpy()
            action_std = np.zeros_like(action_mean)
            finger_std = np.zeros_like(finger_mean)
            applicability_mean = (
                applicability_probs.squeeze(0).detach().cpu().numpy()
                if applicability_probs is not None
                else None
            )
            applicability_std = (
                np.zeros_like(applicability_mean)
                if applicability_mean is not None
                else None
            )
        else:
            mc = _mc_predict(
                self.model,
                x,
                passes=passes,
                temperature_state=self.temperature_state,
                logit_bias_state=self.logit_bias_state,
            )
            action_mean = mc["action_mean"].squeeze(0).detach().cpu().numpy()
            finger_mean = mc["finger_mean"].squeeze(0).detach().cpu().numpy()
            action_std = mc["action_std"].squeeze(0).detach().cpu().numpy()
            finger_std = mc["finger_std"].squeeze(0).detach().cpu().numpy()
            applicability_mean = (
                mc["applicability_mean"].squeeze(0).detach().cpu().numpy()
                if "applicability_mean" in mc
                else None
            )
            applicability_std = (
                mc["applicability_std"].squeeze(0).detach().cpu().numpy()
                if "applicability_std" in mc
                else None
            )

        action_uncertainty = float(np.mean(action_std))
        finger_uncertainty = float(np.mean(finger_std))
        applicability_uncertainty = (
            float(np.mean(applicability_std))
            if applicability_std is not None
            else None
        )

        diagnostics = {
            "health_score": compute_health_score(window_TxC),
            "finger_applicable_prob": (
                float(np.asarray(applicability_mean).reshape(-1)[0])
                if applicability_mean is not None
                else None
            ),
            "applicability_uncertainty": applicability_uncertainty,
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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

        action_id, finger_id = decode_prediction_pair(action_mean, finger_mean)

        action_confidence = float(action_mean[action_id])
        finger_confidence = (
            finger_confidence_for_id(finger_mean, finger_id)
            if finger_mean.size
            else 0.0
        )

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
            "finger_applicable_prob": diagnostics.get("finger_applicable_prob"),
            "applicability_uncertainty": diagnostics.get(
                "applicability_uncertainty"
            ),
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

    def _empty_prediction(
        self, window_TxC: np.ndarray
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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
