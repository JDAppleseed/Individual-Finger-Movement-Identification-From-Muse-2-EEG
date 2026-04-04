from __future__ import annotations

import collections
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.command_shaper import CommandShaper, CommandShaperConfig
from utils.default_recipe import LIVE_INFER_RECIPE_DEFAULTS, PSEUDO_LIVE_RECIPE_DEFAULTS
from utils.inference import InferenceConfig, InferenceEngine
from utils.label_schema import (
    ACTIVE_FINGER_IDS,
    ACTION_NAMES,
    ACTION_REST,
    FINGER_NONE,
    FINGER_NAMES,
    decode_finger_prediction,
    decode_prediction_pair,
    finger_confidence_for_id,
    is_valid_action_finger,
    prediction_pair_diagnostics,
)
from utils.model_outputs import infer_output_dims_from_state_dict, unpack_model_outputs
from utils.postprocess import PostprocessSettings, PostprocessState, postprocess_predictions
from utils.runtime_utils import (
    TemperatureScalingState,
    apply_channel_normalizer,
    apply_temperature_to_logits,
    load_normalizer,
    load_temperature_scaling,
)


@dataclass(frozen=True)
class ActuationDecision:
    finger_id: int
    action_id: int
    prob: float


@dataclass(frozen=True)
class ReplayRuntimeConfig:
    window_sec: float = float(LIVE_INFER_RECIPE_DEFAULTS["window_sec"])
    hop_sec: float = float(LIVE_INFER_RECIPE_DEFAULTS["hop_sec"])
    latency_threshold_ms: float = float(LIVE_INFER_RECIPE_DEFAULTS["latency_threshold_ms"])
    actuation_min_prob: float = float(LIVE_INFER_RECIPE_DEFAULTS["actuation_min_prob"])
    actuation_stability: int = int(LIVE_INFER_RECIPE_DEFAULTS["actuation_stability"])
    actuation_cooldown_ms: int = int(LIVE_INFER_RECIPE_DEFAULTS["actuation_cooldown_ms"])
    actuation_repeat_ms: int = int(LIVE_INFER_RECIPE_DEFAULTS["actuation_repeat_ms"])
    actuation_min_speed: float = float(LIVE_INFER_RECIPE_DEFAULTS["actuation_min_speed"])
    modulate_actuation_speed: bool = bool(LIVE_INFER_RECIPE_DEFAULTS["modulate_actuation_speed"])
    actuation_speed_gamma: float = float(LIVE_INFER_RECIPE_DEFAULTS["actuation_speed_gamma"])
    use_inference_engine: bool = bool(LIVE_INFER_RECIPE_DEFAULTS["use_inference_engine"])
    mc_passes: int = int(LIVE_INFER_RECIPE_DEFAULTS["mc_passes"])
    uncertainty_base_threshold: float = float(
        LIVE_INFER_RECIPE_DEFAULTS["uncertainty_base_threshold"]
    )
    uncertainty_weight: float = float(LIVE_INFER_RECIPE_DEFAULTS["uncertainty_weight"])
    latency_mode: str = str(PSEUDO_LIVE_RECIPE_DEFAULTS["latency_mode"])
    fixed_latency_ms: Optional[float] = PSEUDO_LIVE_RECIPE_DEFAULTS["fixed_latency_ms"]
    reset_on_trial_change: bool = bool(PSEUDO_LIVE_RECIPE_DEFAULTS["reset_on_trial_change"])
    deterministic: bool = bool(PSEUDO_LIVE_RECIPE_DEFAULTS["deterministic"])


def load_train_config(run_dir: Path) -> dict:
    path = Path(run_dir).expanduser().resolve() / "train_config.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def resolve_temperature_path(run_dir: Path) -> Path:
    train_cfg = load_train_config(Path(run_dir))
    candidate = train_cfg.get("save_temperature_path")
    if candidate:
        return Path(str(candidate)).expanduser()
    return Path(run_dir).expanduser().resolve() / "temperature_scaling.json"


def get_deployment_model_info(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    train_cfg = load_train_config(run_dir)
    model_path = run_dir / "finger_action_model.pt"
    temperature_path = resolve_temperature_path(run_dir)
    n_fingers = None
    n_actions = None
    has_applicability_head = None
    if model_path.exists():
        try:
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
            n_fingers, n_actions, has_applicability_head = infer_output_dims_from_state_dict(
                state_dict
            )
        except Exception:
            n_fingers = None
            n_actions = None
            has_applicability_head = None
    active_finger_head = train_cfg.get("active_finger_head")
    finger_applicability_head = train_cfg.get("finger_applicability_head")
    temperature_state = load_temperature_scaling(temperature_path)
    has_applicability_temperature = bool(
        temperature_state is not None
        and temperature_state.has_applicability_temperature
    )
    deployable = bool(
        active_finger_head is True
        and finger_applicability_head is True
        and n_fingers == len(ACTIVE_FINGER_IDS)
        and has_applicability_head is True
        and has_applicability_temperature is True
    )
    reasons: list[str] = []
    if active_finger_head is not True:
        reasons.append("train_config.active_finger_head must be true")
    if finger_applicability_head is not True:
        reasons.append("train_config.finger_applicability_head must be true")
    if n_fingers != len(ACTIVE_FINGER_IDS):
        reasons.append(f"model finger head must have {len(ACTIVE_FINGER_IDS)} outputs")
    if has_applicability_head is not True:
        reasons.append("model must include finger_applicability_head")
    if has_applicability_temperature is not True:
        reasons.append("temperature_scaling.json must include applicability_temperature")
    return {
        "run_dir": str(run_dir),
        "active_finger_head": active_finger_head,
        "finger_applicability_head": finger_applicability_head,
        "n_fingers": n_fingers,
        "n_actions": n_actions,
        "has_applicability_head": has_applicability_head,
        "has_applicability_temperature": has_applicability_temperature,
        "deployable": deployable,
        "reasons": reasons,
    }


def require_deployable_run(run_dir: Path) -> dict[str, Any]:
    info = get_deployment_model_info(run_dir)
    if not info["deployable"]:
        reasons = "; ".join(str(reason) for reason in info["reasons"]) or "unknown"
        raise RuntimeError(
            f"Deployment model requirement failed for {run_dir}: {reasons}"
        )
    return info


def load_model_artifacts(
    *,
    run_dir: Path,
    device: torch.device,
    n_channels: int,
) -> tuple[CNNLSTMFingerActionNet, Any, Optional[TemperatureScalingState]]:
    run_dir = Path(run_dir).expanduser().resolve()
    model_path = run_dir / "finger_action_model.pt"
    scaler_path = run_dir / "scaler.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    normalizer = load_normalizer(scaler_path)
    if normalizer is None:
        raise RuntimeError(f"Failed to load normalizer: {scaler_path}")

    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    n_fingers, n_actions, has_applicability_head = infer_output_dims_from_state_dict(
        state_dict
    )
    model = CNNLSTMFingerActionNet(
        n_channels=int(n_channels),
        n_fingers=n_fingers,
        n_actions=n_actions,
        finger_applicability_head=bool(has_applicability_head),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    temperature_state = load_temperature_scaling(resolve_temperature_path(run_dir))
    return model, normalizer, temperature_state


def is_noop_decision(finger_id: int, action_id: int) -> bool:
    return int(finger_id) == 0 or int(action_id) == 0


def postprocess_decision(
    action_probs: np.ndarray,
    finger_probs: np.ndarray,
    *,
    enabled: bool,
    settings: PostprocessSettings,
    state: PostprocessState,
    finger_applicable_prob: Optional[float] = None,
) -> dict:
    if not enabled:
        raw_action = int(np.argmax(action_probs)) if action_probs.size else 0
        raw_finger = decode_finger_prediction(finger_probs)
        committed_action, committed_finger = decode_prediction_pair(
            action_probs, finger_probs
        )
        action_conf = float(np.max(action_probs)) if action_probs.size else 0.0
        finger_conf = (
            finger_confidence_for_id(finger_probs, committed_finger)
            if finger_probs.size
            else 0.0
        )
        finger_gate_ok = bool(
            committed_action == int(ACTION_REST)
            or finger_conf >= float(settings.threshold_finger)
        )
        applicability_gate_ok = bool(
            committed_action == int(ACTION_REST)
            or finger_applicable_prob is None
            or float(finger_applicable_prob) >= float(settings.threshold_applicability)
        )
        return {
            "committed_action_id": committed_action,
            "committed_finger_id": committed_finger,
            "raw_top_action_id": raw_action,
            "raw_top_finger_id": raw_finger,
            "action_conf": action_conf,
            "finger_conf": finger_conf,
            "finger_gate_ok": finger_gate_ok,
            "finger_applicable_prob": (
                float(finger_applicable_prob)
                if finger_applicable_prob is not None
                else None
            ),
            "applicability_gate_ok": applicability_gate_ok,
            "committed_pair_valid": bool(
                is_valid_action_finger(committed_action, committed_finger)
            ),
            "smoothed_action_id": committed_action,
            "smoothed_finger_id": committed_finger,
            "decision_reason": "raw_argmax_gated",
            "frames_in_state": 1,
        }
    return postprocess_predictions(
        action_probs,
        finger_probs,
        settings,
        state,
        finger_applicable_prob=finger_applicable_prob,
    )


def predict_window(
    window: np.ndarray,
    *,
    scaler: object,
    model: torch.nn.Module,
    device: torch.device,
    inference_engine: Optional[InferenceEngine],
    direct_engine: Optional[InferenceEngine] = None,
    temperature_state: Optional[TemperatureScalingState] = None,
) -> dict[str, Any]:
    window_f32 = np.asarray(window, dtype=np.float32)

    if inference_engine is None:
        if direct_engine is not None:
            _, x = direct_engine.prepare_input(window_f32)
            finger_probs_t, action_probs_t, applicability_prob = (
                direct_engine.forward_probabilities(x)
            )
            action_probs_t = action_probs_t.squeeze(0)
            finger_probs_t = finger_probs_t.squeeze(0)
            applicability_prob = (
                applicability_prob.squeeze(0)
                if applicability_prob is not None
                else None
            )
        else:
            window_input = apply_channel_normalizer(window_f32, scaler)
            x = torch.from_numpy(window_input).unsqueeze(0).to(device)
            with torch.inference_mode():
                finger_logits, action_logits, applicability_logits = unpack_model_outputs(
                    model(x)
                )
                finger_logits = apply_temperature_to_logits(
                    finger_logits,
                    temperature_state.finger_temperature
                    if temperature_state is not None
                    else 1.0,
                )
                action_logits = apply_temperature_to_logits(
                    action_logits,
                    temperature_state.action_temperature
                    if temperature_state is not None
                    else 1.0,
                )
                if applicability_logits is not None:
                    applicability_logits = apply_temperature_to_logits(
                        applicability_logits,
                        temperature_state.applicability_temperature
                        if temperature_state is not None
                        else 1.0,
                    )
                action_probs_t = torch.softmax(action_logits, dim=1).squeeze(0)
                finger_probs_t = torch.softmax(finger_logits, dim=1).squeeze(0)
                applicability_prob = (
                    torch.sigmoid(applicability_logits).squeeze(0)
                    if applicability_logits is not None
                    else None
                )
        return {
            "backend": "direct",
            "action_probs": action_probs_t.detach().cpu().numpy(),
            "finger_probs": finger_probs_t.detach().cpu().numpy(),
            "finger_applicable_prob": (
                float(applicability_prob.detach().cpu().item())
                if applicability_prob is not None
                else None
            ),
            "action_uncertainty": 0.0,
            "finger_uncertainty": 0.0,
            "applicability_uncertainty": None,
            "adaptive_threshold": None,
            "health_score": None,
        }

    (
        action_probs,
        finger_probs,
        action_uncertainty,
        finger_uncertainty,
        diagnostics,
    ) = inference_engine.predict_proba(window_f32)
    if action_probs is None or finger_probs is None:
        raise RuntimeError(
            "InferenceEngine returned empty probabilities for a loaded model."
        )
    adaptive_threshold = min(
        0.99,
        max(
            float(inference_engine.config.base_threshold),
            float(inference_engine.config.base_threshold)
            + float(inference_engine.config.uncertainty_weight)
            * float(action_uncertainty),
        ),
    )
    return {
        "backend": "inference_engine",
        "action_probs": action_probs,
        "finger_probs": finger_probs,
        "finger_applicable_prob": diagnostics.get("finger_applicable_prob"),
        "action_uncertainty": float(action_uncertainty),
        "finger_uncertainty": float(finger_uncertainty),
        "applicability_uncertainty": diagnostics.get("applicability_uncertainty"),
        "adaptive_threshold": float(adaptive_threshold),
        "health_score": (
            diagnostics.get("health_score") if isinstance(diagnostics, dict) else None
        ),
    }


def debounced_should_send(
    decision: ActuationDecision,
    *,
    last_sent: Optional[Tuple[int, int]],
    stable_count: int,
    required_stability: int,
    last_send_time_ms: Optional[float],
    current_time_ms: float,
    cooldown_ms: int,
    repeat_same_ms: int = 0,
) -> bool:
    if decision.prob <= 0.0:
        return False
    if int(decision.finger_id) == 0 or int(decision.action_id) == 0:
        return False
    if int(stable_count) < int(required_stability):
        return False
    if last_send_time_ms is None:
        elapsed_ms = float("inf")
    else:
        elapsed_ms = float(current_time_ms) - float(last_send_time_ms)
    if last_sent is not None and (decision.finger_id, decision.action_id) == last_sent:
        return elapsed_ms >= float(max(0, int(repeat_same_ms)))
    if elapsed_ms < float(cooldown_ms):
        return False
    return True


def uncertainty_gate_passed(
    decision_info: dict[str, Any],
    inference_result: dict[str, Any],
) -> bool:
    adaptive_threshold = inference_result.get("adaptive_threshold")
    if adaptive_threshold is None:
        return True
    return float(decision_info.get("action_conf", 0.0)) >= float(adaptive_threshold)


def finger_gate_passed(decision_info: dict[str, Any]) -> bool:
    return bool(decision_info.get("finger_gate_ok", True))


def applicability_gate_passed(decision_info: dict[str, Any]) -> bool:
    return bool(decision_info.get("applicability_gate_ok", True))


def build_actuation_speed_mapper(
    *,
    modulate_actuation_speed: bool,
    actuation_speed_gamma: float,
) -> Optional[CommandShaper]:
    if not bool(modulate_actuation_speed):
        return None
    return CommandShaper(
        CommandShaperConfig(base_conf_thresh=0.0, speed_gamma=float(actuation_speed_gamma))
    )


def compute_actuation_speed_scalar(
    decision_prob: float,
    action_uncertainty: float,
    speed_mapper: Optional[CommandShaper],
    *,
    min_speed: float = 0.0,
) -> float:
    confidence = float(max(0.0, min(1.0, decision_prob)))
    confidence *= max(0.0, 1.0 - float(action_uncertainty))
    if speed_mapper is None:
        return 1.0
    speed = float(speed_mapper.confidence_to_speed(confidence))
    min_speed = float(max(0.0, min(1.0, min_speed)))
    if speed > 0.0 and min_speed > 0.0:
        speed = max(min_speed, speed)
    return speed


def build_actuation_command_shaper(
    *,
    actuation_min_prob: float,
    actuation_speed_gamma: float,
    hop_sec: float,
    actuation_stability: int,
    actuation_cooldown_ms: int,
) -> CommandShaper:
    immediate_mode = (
        float(actuation_min_prob) <= 0.0
        and int(actuation_stability) <= 2
        and int(actuation_cooldown_ms) <= 0
    )
    if immediate_mode:
        hold_ms = 0
        hold_conf_margin = 0.0
    else:
        min_hold_ms = max(
            int(actuation_cooldown_ms),
            int(round(float(hop_sec) * 1000.0 * max(1, int(actuation_stability)))),
        )
        hold_ms = max(150, min_hold_ms)
        hold_conf_margin = 0.05
    return CommandShaper(
        CommandShaperConfig(
            base_conf_thresh=float(actuation_min_prob),
            speed_gamma=float(actuation_speed_gamma),
            hold_ms=int(hold_ms),
            hold_conf_margin=float(hold_conf_margin),
        )
    )


def latency_gate_passed(latency_ms: float, threshold_ms: float) -> bool:
    latency_ms = float(latency_ms)
    threshold_ms = float(threshold_ms)
    return -50.0 <= latency_ms <= threshold_ms


def resolve_actuation_candidate(
    history: Deque[ActuationDecision],
    *,
    required_finger_stability: int,
) -> dict[str, Any]:
    required = max(1, int(required_finger_stability))
    if len(history) < required:
        return {
            "decision": ActuationDecision(finger_id=0, action_id=0, prob=0.0),
            "reason": "pair_stability",
            "finger_votes": {},
            "action_votes": {},
            "pair_votes": {},
            "resolved_finger_id": 0,
        }

    tail = list(history)[-required:]
    finger_ids = [int(d.finger_id) for d in tail]
    action_ids = [int(d.action_id) for d in tail]
    pair_ids = [(int(d.finger_id), int(d.action_id)) for d in tail]
    pair_counts = collections.Counter(pair_ids)
    nonzero_pairs = [
        pair for pair in pair_ids if int(pair[0]) != 0 and int(pair[1]) != 0
    ]
    if len(nonzero_pairs) != required or len(set(nonzero_pairs)) != 1:
        return {
            "decision": ActuationDecision(finger_id=0, action_id=0, prob=0.0),
            "reason": "pair_stability",
            "finger_votes": dict(collections.Counter(finger_ids)),
            "action_votes": dict(collections.Counter(action_ids)),
            "pair_votes": {
                f"{int(fid)}:{int(aid)}": int(count)
                for (fid, aid), count in pair_counts.items()
            },
            "resolved_finger_id": 0,
        }

    resolved_finger_id, chosen_action_id = nonzero_pairs[0]
    chosen_prob = float(np.mean([float(d.prob) for d in tail])) if tail else 0.0
    return {
        "decision": ActuationDecision(
            finger_id=int(resolved_finger_id),
            action_id=int(chosen_action_id),
            prob=chosen_prob,
        ),
        "reason": "exact_pair_stability",
        "finger_votes": dict(collections.Counter(finger_ids)),
        "action_votes": dict(collections.Counter(action_ids)),
        "pair_votes": {
            f"{int(fid)}:{int(aid)}": int(count)
            for (fid, aid), count in pair_counts.items()
        },
        "resolved_finger_id": int(resolved_finger_id),
    }


def build_inference_engine(
    *,
    model: torch.nn.Module,
    scaler: object,
    device: torch.device,
    runtime_config: ReplayRuntimeConfig,
    temperature_state: Optional[TemperatureScalingState],
) -> Optional[InferenceEngine]:
    if not bool(runtime_config.use_inference_engine):
        return None
    config = InferenceConfig(
        base_threshold=float(runtime_config.uncertainty_base_threshold),
        uncertainty_weight=float(runtime_config.uncertainty_weight),
        stability_frames=max(1, int(runtime_config.actuation_stability)),
        mc_passes=max(1, int(runtime_config.mc_passes)),
    )
    return InferenceEngine(
        model=model,
        normalizer=scaler,
        device=device,
        action_names={},
        finger_names={},
        config=config,
        temperature_state=temperature_state,
    )


def estimate_prediction_latency_ms(
    *,
    window_sec: float,
    offline_compute_ms: float,
    latency_mode: str,
    fixed_latency_ms: Optional[float],
) -> float:
    mode = str(latency_mode or "ignore").strip().lower()
    if mode == "fixed":
        if fixed_latency_ms is None:
            raise ValueError("fixed latency mode requires fixed_latency_ms")
        return float(fixed_latency_ms)
    return float(max(0.0, float(window_sec) * 500.0 + float(offline_compute_ms)))


def _set_deterministic_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def replay_ordered_windows(
    *,
    X: np.ndarray,
    window_start_s: np.ndarray,
    window_end_s: np.ndarray,
    y_action_true: np.ndarray,
    y_finger_true: np.ndarray,
    trial_ids: Optional[np.ndarray],
    session_ids: Optional[np.ndarray],
    event_ids: Optional[np.ndarray],
    event_onset_s: Optional[np.ndarray],
    scaler: object,
    model: torch.nn.Module,
    device: torch.device,
    postprocess_enabled: bool,
    postprocess_settings: PostprocessSettings,
    runtime_config: ReplayRuntimeConfig,
    temperature_state: Optional[TemperatureScalingState] = None,
) -> dict[str, Any]:
    if runtime_config.deterministic:
        _set_deterministic_seed(0)

    inference_engine = build_inference_engine(
        model=model,
        scaler=scaler,
        device=device,
        runtime_config=runtime_config,
        temperature_state=temperature_state,
    )
    direct_engine = (
        None
        if inference_engine is not None
        else InferenceEngine(
            model=model,
            normalizer=scaler,
            device=device,
            action_names={},
            finger_names={},
            config=InferenceConfig(mc_passes=1),
            temperature_state=temperature_state,
        )
    )
    post_state = PostprocessState()
    actuation_history: Deque[ActuationDecision] = collections.deque(
        maxlen=max(3, int(runtime_config.actuation_stability))
    )
    actuation_speed_mapper = build_actuation_speed_mapper(
        modulate_actuation_speed=bool(runtime_config.modulate_actuation_speed),
        actuation_speed_gamma=float(runtime_config.actuation_speed_gamma),
    )
    actuation_command_shaper = build_actuation_command_shaper(
        actuation_min_prob=float(runtime_config.actuation_min_prob),
        actuation_speed_gamma=float(runtime_config.actuation_speed_gamma),
        hop_sec=float(runtime_config.hop_sec),
        actuation_stability=int(runtime_config.actuation_stability),
        actuation_cooldown_ms=int(runtime_config.actuation_cooldown_ms),
    )

    records: List[Dict[str, Any]] = []
    last_sent: Optional[Tuple[int, int]] = None
    last_send_time_ms: Optional[float] = None
    last_trial: Optional[int] = None

    n = int(len(X))
    for idx in range(n):
        if trial_ids is not None and bool(runtime_config.reset_on_trial_change):
            current_trial = int(trial_ids[idx])
            if last_trial is None:
                last_trial = current_trial
            elif current_trial != last_trial:
                post_state.reset()
                actuation_history.clear()
                actuation_command_shaper.reset()
                last_sent = None
                last_send_time_ms = None
                last_trial = current_trial

        loop_start = time.perf_counter()
        inference_result = predict_window(
            X[idx],
            scaler=scaler,
            model=model,
            device=device,
            inference_engine=inference_engine,
            direct_engine=direct_engine,
            temperature_state=temperature_state,
        )
        action_probs = np.asarray(inference_result["action_probs"], dtype=float)
        finger_probs = np.asarray(inference_result["finger_probs"], dtype=float)
        finger_applicable_prob = inference_result.get("finger_applicable_prob")
        action_uncertainty = float(inference_result.get("action_uncertainty", 0.0) or 0.0)
        finger_uncertainty = float(inference_result.get("finger_uncertainty", 0.0) or 0.0)
        applicability_uncertainty = inference_result.get("applicability_uncertainty")

        decision_info = postprocess_decision(
            action_probs,
            finger_probs,
            enabled=bool(postprocess_enabled),
            settings=postprocess_settings,
            state=post_state,
            finger_applicable_prob=(
                float(finger_applicable_prob)
                if finger_applicable_prob is not None
                else None
            ),
        )
        decision = ActuationDecision(
            finger_id=int(decision_info["committed_finger_id"]),
            action_id=int(decision_info["committed_action_id"]),
            prob=float(
                min(
                    float(decision_info.get("action_conf", 0.0)),
                    float(decision_info.get("finger_conf", 0.0)),
                )
            ),
        )
        finger_gate_ok = finger_gate_passed(decision_info)
        applicability_gate_ok = applicability_gate_passed(decision_info)
        uncertainty_gate_ok = uncertainty_gate_passed(decision_info, inference_result)
        actuation_speed_scalar = compute_actuation_speed_scalar(
            decision.prob,
            action_uncertainty,
            actuation_speed_mapper,
            min_speed=float(runtime_config.actuation_min_speed),
        )

        window_start = float(window_start_s[idx])
        window_end = float(window_end_s[idx])
        window_center_stream_s = window_start + (window_end - window_start) / 2.0
        current_time_ms = float(window_center_stream_s * 1000.0)

        actuation_history.append(decision)
        actuation_vote = resolve_actuation_candidate(
            actuation_history,
            required_finger_stability=int(runtime_config.actuation_stability),
        )
        voted_decision = actuation_vote["decision"]
        actuation_target_finger_id = int(voted_decision.finger_id)
        actuation_target_action_id = int(voted_decision.action_id)
        actuation_suppressed_reason: Optional[str] = None
        actuation_sent = False
        actuation_decision_delay_ms = None

        loop_end = time.perf_counter()
        offline_compute_ms = float((loop_end - loop_start) * 1000.0)
        prediction_latency_ms = estimate_prediction_latency_ms(
            window_sec=float(runtime_config.window_sec),
            offline_compute_ms=offline_compute_ms,
            latency_mode=str(runtime_config.latency_mode),
            fixed_latency_ms=runtime_config.fixed_latency_ms,
        )
        if str(runtime_config.latency_mode).strip().lower() == "ignore":
            actuation_latency_gate_ok = True
        else:
            actuation_latency_gate_ok = latency_gate_passed(
                prediction_latency_ms,
                float(runtime_config.latency_threshold_ms),
            )

        if not actuation_latency_gate_ok:
            actuation_suppressed_reason = "latency_gate"
        elif not applicability_gate_ok:
            actuation_suppressed_reason = "applicability_gate"
        elif not finger_gate_ok:
            actuation_suppressed_reason = "finger_gate"
        elif is_noop_decision(voted_decision.finger_id, voted_decision.action_id):
            actuation_suppressed_reason = str(actuation_vote.get("reason", "noop"))
        elif not uncertainty_gate_ok:
            actuation_suppressed_reason = "uncertainty_gate"
        else:
            shaped_command = actuation_command_shaper.shape(
                action_id=int(voted_decision.action_id),
                finger_id=int(voted_decision.finger_id),
                action_conf=float(voted_decision.prob),
                speed_scalar_override=float(actuation_speed_scalar),
                timestamp_stream_ms=int(round(window_center_stream_s * 1000.0)),
                stability_ok=True,
                timebase_ms=int(round(window_center_stream_s * 1000.0)),
            )
            actuation_target_finger_id = int(shaped_command.finger_id)
            actuation_target_action_id = int(shaped_command.action_id)
            actuation_speed_scalar = float(shaped_command.speed_scalar)
            actuation_decision = ActuationDecision(
                finger_id=actuation_target_finger_id,
                action_id=actuation_target_action_id,
                prob=float(voted_decision.prob),
            )
            actuation_key = (
                int(actuation_decision.finger_id),
                int(actuation_decision.action_id),
            )
            if is_noop_decision(
                actuation_decision.finger_id, actuation_decision.action_id
            ):
                actuation_suppressed_reason = "min_prob"
            elif debounced_should_send(
                actuation_decision,
                last_sent=last_sent,
                stable_count=1,
                required_stability=1,
                last_send_time_ms=last_send_time_ms,
                current_time_ms=current_time_ms,
                cooldown_ms=int(runtime_config.actuation_cooldown_ms),
                repeat_same_ms=int(runtime_config.actuation_repeat_ms),
            ):
                last_sent = actuation_key
                last_send_time_ms = current_time_ms
                actuation_sent = True
                actuation_decision_delay_ms = 0.0
            else:
                actuation_suppressed_reason = "cooldown_or_duplicate"

        payload = {
            "ts_utc": float(window_end),
            "window_start_s": window_start,
            "window_end_s": window_end,
            "latency_ms": float(prediction_latency_ms),
            "prediction_latency_ms": float(prediction_latency_ms),
            "offline_compute_ms": float(offline_compute_ms),
            "alignment_ok": True,
            "action_probs": action_probs.tolist(),
            "finger_probs": finger_probs.tolist(),
            "raw_top_action_id": int(decision_info.get("raw_top_action_id", 0)),
            "raw_top_finger_id": int(decision_info.get("raw_top_finger_id", 0)),
            "smoothed_action_id": int(decision_info.get("smoothed_action_id", 0)),
            "smoothed_finger_id": int(decision_info.get("smoothed_finger_id", 0)),
            "committed_action_id": int(decision_info.get("committed_action_id", 0)),
            "committed_finger_id": int(decision_info.get("committed_finger_id", 0)),
            "action_conf": float(decision_info.get("action_conf", 0.0)),
            "finger_conf": float(decision_info.get("finger_conf", 0.0)),
            "finger_gate_ok": bool(decision_info.get("finger_gate_ok", True)),
            "finger_applicable_prob": decision_info.get("finger_applicable_prob"),
            "applicability_gate_ok": bool(
                decision_info.get("applicability_gate_ok", True)
            ),
            "committed_pair_valid": bool(
                decision_info.get("committed_pair_valid", True)
            ),
            "joint_conf": float(decision.prob),
            "action_uncertainty": action_uncertainty,
            "finger_uncertainty": finger_uncertainty,
            "applicability_uncertainty": applicability_uncertainty,
            "adaptive_threshold": inference_result.get("adaptive_threshold"),
            "uncertainty_gate_ok": bool(uncertainty_gate_ok),
            "health_score": inference_result.get("health_score"),
            "inference_backend": str(inference_result.get("backend", "direct")),
            "decision_reason": str(decision_info.get("decision_reason", "")),
            "postprocess_enabled": bool(postprocess_enabled),
            "dropped_windows": 0,
            "actuation_speed_scalar": float(actuation_speed_scalar),
            "actuation_target_finger_id": int(actuation_target_finger_id),
            "actuation_target_action_id": int(actuation_target_action_id),
            "actuation_vote_reason": str(actuation_vote.get("reason", "")),
            "actuation_vote_finger_counts": actuation_vote.get("finger_votes", {}),
            "actuation_vote_action_counts": actuation_vote.get("action_votes", {}),
            "actuation_latency_gate_ok": bool(actuation_latency_gate_ok),
            "actuation_suppressed_reason": actuation_suppressed_reason,
            "actuation_sent": bool(actuation_sent),
            "actuation_latency_ms": (
                float(prediction_latency_ms) if bool(actuation_sent) else None
            ),
            "actuation_decision_delay_ms": (
                float(actuation_decision_delay_ms)
                if actuation_decision_delay_ms is not None
                else None
            ),
            "latency_mode": str(runtime_config.latency_mode),
        }
        if session_ids is not None:
            payload["session_id"] = str(session_ids[idx])
        if event_ids is not None:
            payload["event_id"] = int(event_ids[idx])
        if event_onset_s is not None:
            payload["event_onset_s"] = float(event_onset_s[idx])
        if trial_ids is not None:
            payload["trial_id"] = int(trial_ids[idx])
        payload["true_action_id"] = int(y_action_true[idx])
        payload["true_finger_id"] = int(y_finger_true[idx])
        records.append(payload)

    return {
        "records": records,
        "runtime_config": asdict(runtime_config),
        "postprocess_enabled": bool(postprocess_enabled),
        "postprocess_settings": asdict(postprocess_settings),
    }


def write_predictions_jsonl(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")


def _series_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _contiguous_segments(
    *,
    action_ids: np.ndarray,
    finger_ids: np.ndarray,
    window_start_s: np.ndarray,
    window_end_s: np.ndarray,
    session_ids: Optional[np.ndarray],
    trial_ids: Optional[np.ndarray],
    non_rest_only: bool = True,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for idx in range(len(action_ids)):
        action_id = int(action_ids[idx])
        finger_id = int(finger_ids[idx])
        if non_rest_only and is_noop_decision(finger_id, action_id):
            pair = None
        else:
            pair = (action_id, finger_id)
        session_id = (
            str(session_ids[idx]) if session_ids is not None else "unknown_session"
        )
        trial_id = int(trial_ids[idx]) if trial_ids is not None else -1
        if pair is None:
            if current is not None:
                segments.append(current)
                current = None
            continue
        if (
            current is None
            or tuple(current["pair"]) != pair
            or current["session_id"] != session_id
            or int(current["trial_id"]) != trial_id
        ):
            if current is not None:
                segments.append(current)
            current = {
                "pair": [int(pair[0]), int(pair[1])],
                "pair_label": f"{ACTION_NAMES.get(pair[0], pair[0])}+{FINGER_NAMES.get(pair[1], pair[1])}",
                "action_id": int(pair[0]),
                "finger_id": int(pair[1]),
                "session_id": session_id,
                "trial_id": int(trial_id),
                "start_s": float(window_start_s[idx]),
                "end_s": float(window_end_s[idx]),
                "window_count": 1,
            }
        else:
            current["end_s"] = float(window_end_s[idx])
            current["window_count"] = int(current["window_count"]) + 1
    if current is not None:
        segments.append(current)
    return segments


def _segment_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = max(0.0, min(float(a["end_s"]), float(b["end_s"])) - max(float(a["start_s"]), float(b["start_s"])))
    union = max(float(a["end_s"]), float(b["end_s"])) - min(float(a["start_s"]), float(b["start_s"]))
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _segment_overlap_summary(
    *,
    predicted_action: np.ndarray,
    predicted_finger: np.ndarray,
    true_action: np.ndarray,
    true_finger: np.ndarray,
    window_start_s: np.ndarray,
    window_end_s: np.ndarray,
    session_ids: Optional[np.ndarray],
    trial_ids: Optional[np.ndarray],
) -> Dict[str, Any]:
    pred_segments = _contiguous_segments(
        action_ids=predicted_action,
        finger_ids=predicted_finger,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        session_ids=session_ids,
        trial_ids=trial_ids,
        non_rest_only=True,
    )
    true_segments = _contiguous_segments(
        action_ids=true_action,
        finger_ids=true_finger,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        session_ids=session_ids,
        trial_ids=trial_ids,
        non_rest_only=True,
    )
    best_ious: List[float] = []
    matched_025 = 0
    matched_050 = 0
    for pred in pred_segments:
        candidates = [
            truth
            for truth in true_segments
            if truth["pair"] == pred["pair"]
            and truth["session_id"] == pred["session_id"]
            and int(truth["trial_id"]) == int(pred["trial_id"])
        ]
        best = max((_segment_iou(pred, truth) for truth in candidates), default=0.0)
        best_ious.append(float(best))
        if best >= 0.25:
            matched_025 += 1
        if best >= 0.50:
            matched_050 += 1
    return {
        "predicted_segment_count": int(len(pred_segments)),
        "truth_segment_count": int(len(true_segments)),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else None,
        "median_best_iou": float(np.median(best_ious)) if best_ious else None,
        "match_rate_iou_0_25": (
            float(matched_025 / len(pred_segments)) if pred_segments else None
        ),
        "match_rate_iou_0_50": (
            float(matched_050 / len(pred_segments)) if pred_segments else None
        ),
    }


def _first_match_latency_summary(
    *,
    match_mask: np.ndarray,
    event_ids: Optional[np.ndarray],
    session_ids: Optional[np.ndarray],
    event_onset_s: Optional[np.ndarray],
    fallback_window_start_s: np.ndarray,
    detection_time_s: np.ndarray,
    true_action: np.ndarray,
) -> Dict[str, Any]:
    if event_ids is None:
        return {"count": 0, "mean": None, "median": None, "p95": None, "miss_count": 0}
    latencies: List[float] = []
    miss_count = 0
    keys_seen = set()
    for idx in range(len(event_ids)):
        if int(true_action[idx]) == int(ACTION_REST):
            continue
        event_id = int(event_ids[idx])
        if event_id < 0:
            continue
        session_id = str(session_ids[idx]) if session_ids is not None else "unknown_session"
        key = (session_id, event_id)
        if key in keys_seen:
            continue
        keys_seen.add(key)
        event_mask = (
            (np.asarray(event_ids, dtype=np.int64) == event_id)
            & (np.asarray(true_action, dtype=np.int64) != int(ACTION_REST))
        )
        if session_ids is not None:
            event_mask &= np.asarray(session_ids).astype("U") == session_id
        onset_candidates = []
        if event_onset_s is not None:
            onset_candidates = [
                float(v)
                for v in np.asarray(event_onset_s)[event_mask]
                if np.isfinite(float(v))
            ]
        onset_s = (
            min(onset_candidates)
            if onset_candidates
            else float(np.min(np.asarray(fallback_window_start_s)[event_mask]))
        )
        detected = detection_time_s[event_mask & match_mask]
        if detected.size == 0:
            miss_count += 1
            continue
        first_detect = float(np.min(detected))
        latencies.append(float(first_detect - onset_s))
    out = _series_summary(latencies)
    out["miss_count"] = int(miss_count)
    return out


def compute_replay_metrics(
    *,
    records: List[Dict[str, Any]],
    y_action_true: np.ndarray,
    y_finger_true: np.ndarray,
    window_start_s: np.ndarray,
    window_end_s: np.ndarray,
    trial_ids: Optional[np.ndarray],
    session_ids: Optional[np.ndarray],
    event_ids: Optional[np.ndarray],
    event_onset_s: Optional[np.ndarray],
    applicability_threshold: float = 0.5,
) -> Dict[str, Any]:
    committed_action = np.asarray(
        [int(row.get("committed_action_id", 0)) for row in records], dtype=np.int64
    )
    committed_finger = np.asarray(
        [int(row.get("committed_finger_id", 0)) for row in records], dtype=np.int64
    )
    actuation_sent = np.asarray(
        [bool(row.get("actuation_sent", False)) for row in records], dtype=bool
    )
    actuation_action = np.asarray(
        [int(row.get("actuation_target_action_id", 0)) for row in records], dtype=np.int64
    )
    actuation_finger = np.asarray(
        [int(row.get("actuation_target_finger_id", 0)) for row in records], dtype=np.int64
    )
    committed_pair_valid = np.asarray(
        [bool(row.get("committed_pair_valid", True)) for row in records], dtype=bool
    )
    applicability_gate_ok = np.asarray(
        [bool(row.get("applicability_gate_ok", True)) for row in records], dtype=bool
    )
    finger_applicable_prob_values = [
        row.get("finger_applicable_prob") for row in records
    ]
    has_applicability_prob = any(value is not None for value in finger_applicable_prob_values)
    finger_applicable_prob = (
        np.asarray(
            [
                float(value) if value is not None else np.nan
                for value in finger_applicable_prob_values
            ],
            dtype=float,
        )
        if has_applicability_prob
        else None
    )
    offline_compute_ms = np.asarray(
        [float(row.get("offline_compute_ms", 0.0) or 0.0) for row in records],
        dtype=float,
    )

    y_action_true = np.asarray(y_action_true, dtype=np.int64)
    y_finger_true = np.asarray(y_finger_true, dtype=np.int64)
    non_rest_mask = y_action_true != int(ACTION_REST)
    rest_mask = y_action_true == int(ACTION_REST)

    committed_joint_correct = (committed_action == y_action_true) & (
        committed_finger == y_finger_true
    )
    would_send_match = actuation_sent & (actuation_action == y_action_true) & (
        actuation_finger == y_finger_true
    ) & non_rest_mask
    positive_send_mask = actuation_sent & (
        (actuation_action != int(ACTION_REST)) & (actuation_finger != 0)
    )
    sent_action_effective = np.where(
        actuation_sent, actuation_action, int(ACTION_REST)
    ).astype(np.int64)
    sent_finger_effective = np.where(
        actuation_sent, actuation_finger, int(FINGER_NONE)
    ).astype(np.int64)
    pair_diag = prediction_pair_diagnostics(
        committed_action,
        committed_finger,
        committed_action_ids=committed_action,
        committed_finger_ids=committed_finger,
        sent_action_ids=sent_action_effective,
        sent_finger_ids=sent_finger_effective,
    )

    committed_detection_time_s = np.asarray(window_end_s, dtype=float)
    would_send_time_s = np.asarray(window_end_s, dtype=float) + (
        offline_compute_ms / 1000.0
    )
    applicability_fp_rate_on_true_rest = None
    applicability_fn_rate_on_true_non_rest = None
    action_applicability_disagreement_rate = None
    if finger_applicable_prob is not None:
        valid_applicability_mask = np.isfinite(finger_applicable_prob)
        predicted_applicable = np.asarray(
            np.nan_to_num(finger_applicable_prob, nan=0.0)
            >= float(applicability_threshold),
            dtype=bool,
        )
        rest_mask_valid = rest_mask & valid_applicability_mask
        non_rest_mask_valid = non_rest_mask & valid_applicability_mask
        if np.any(rest_mask_valid):
            applicability_fp_rate_on_true_rest = float(
                np.mean(predicted_applicable[rest_mask_valid])
            )
        if np.any(non_rest_mask_valid):
            applicability_fn_rate_on_true_non_rest = float(
                np.mean(~predicted_applicable[non_rest_mask_valid])
            )
        if np.any(valid_applicability_mask):
            action_applicability_disagreement_rate = float(
                np.mean(
                    predicted_applicable[valid_applicability_mask]
                    != (
                        committed_action[valid_applicability_mask]
                        != int(ACTION_REST)
                    )
                )
            )

    return {
        "committed_action_acc": float(accuracy_score(y_action_true, committed_action))
        if y_action_true.size
        else None,
        "committed_joint_acc": float(np.mean(committed_joint_correct))
        if committed_joint_correct.size
        else None,
        "committed_finger_acc_non_rest": float(
            accuracy_score(y_finger_true[non_rest_mask], committed_finger[non_rest_mask])
        )
        if np.any(non_rest_mask)
        else None,
        "would_send_window_precision_non_rest": float(
            np.sum(would_send_match) / np.sum(positive_send_mask)
        )
        if np.any(positive_send_mask)
        else None,
        "would_send_window_recall_non_rest": float(
            np.sum(would_send_match) / np.sum(non_rest_mask)
        )
        if np.any(non_rest_mask)
        else None,
        "false_actuation_rate_rest": float(
            np.sum(actuation_sent & rest_mask) / np.sum(rest_mask)
        )
        if np.any(rest_mask)
        else None,
        "non_rest_none_count": int(pair_diag["committed_non_rest_none_count"]),
        "committed_non_rest_none_count": int(pair_diag["committed_non_rest_none_count"]),
        "committed_non_rest_none_rate": pair_diag["committed_non_rest_none_rate"],
        "committed_rest_non_none_count": int(pair_diag["committed_rest_non_none_count"]),
        "committed_rest_non_none_rate": pair_diag["committed_rest_non_none_rate"],
        "sent_non_rest_none_count": int(pair_diag["sent_non_rest_none_count"]),
        "sent_non_rest_none_rate": pair_diag["sent_non_rest_none_rate"],
        "sent_rest_non_none_count": int(pair_diag["sent_rest_non_none_count"]),
        "sent_rest_non_none_rate": pair_diag["sent_rest_non_none_rate"],
        "applicability_fp_rate_on_true_rest": applicability_fp_rate_on_true_rest,
        "applicability_fn_rate_on_true_non_rest": applicability_fn_rate_on_true_non_rest,
        "action_applicability_disagreement_rate": action_applicability_disagreement_rate,
        "deployment_pair_invariant_ok": bool(
            pair_diag["committed_non_rest_none_count"] == 0
            and pair_diag["committed_rest_non_none_count"] == 0
            and bool(np.all(committed_pair_valid))
            and pair_diag["sent_non_rest_none_count"] == 0
            and pair_diag["sent_rest_non_none_count"] == 0
        ),
        "committed_first_onset_latency_s": _first_match_latency_summary(
            match_mask=committed_joint_correct,
            event_ids=event_ids,
            session_ids=session_ids,
            event_onset_s=event_onset_s,
            fallback_window_start_s=np.asarray(window_start_s, dtype=float),
            detection_time_s=committed_detection_time_s,
            true_action=y_action_true,
        ),
        "would_send_first_onset_latency_s": _first_match_latency_summary(
            match_mask=would_send_match,
            event_ids=event_ids,
            session_ids=session_ids,
            event_onset_s=event_onset_s,
            fallback_window_start_s=np.asarray(window_start_s, dtype=float),
            detection_time_s=would_send_time_s,
            true_action=y_action_true,
        ),
        "committed_segment_overlap": _segment_overlap_summary(
            predicted_action=committed_action,
            predicted_finger=committed_finger,
            true_action=y_action_true,
            true_finger=y_finger_true,
            window_start_s=np.asarray(window_start_s, dtype=float),
            window_end_s=np.asarray(window_end_s, dtype=float),
            session_ids=session_ids,
            trial_ids=trial_ids,
        ),
    }
