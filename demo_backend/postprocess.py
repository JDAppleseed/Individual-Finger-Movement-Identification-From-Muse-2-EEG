from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np


@dataclass
class PostprocessSettings:
    smoothing_enabled: bool = True
    smoothing_method: str = "vote"  # "vote" or "ema"
    smoothing_window: int = 5
    hysteresis_enabled: bool = True
    hysteresis_frames: int = 3
    threshold_action: float = 0.75
    threshold_finger: float = 0.75
    adjacency_enabled: bool = True
    hysteresis_margin: float = 0.05
    finger_delta: float = 0.05


@dataclass
class PostprocessState:
    action_ids: Deque[int] = field(default_factory=deque)
    finger_ids: Deque[int] = field(default_factory=deque)
    action_probs: Deque[np.ndarray] = field(default_factory=deque)
    finger_probs: Deque[np.ndarray] = field(default_factory=deque)
    ema_action: Optional[np.ndarray] = None
    ema_finger: Optional[np.ndarray] = None
    last_action: int = 0
    last_finger: int = 0
    pending_action: Optional[int] = None
    pending_count: int = 0
    frames_in_state: int = 0

    def reset(self) -> None:
        self.action_ids.clear()
        self.finger_ids.clear()
        self.action_probs.clear()
        self.finger_probs.clear()
        self.ema_action = None
        self.ema_finger = None
        self.last_action = 0
        self.last_finger = 0
        self.pending_action = None
        self.pending_count = 0
        self.frames_in_state = 0


def _vote_majority(ids: Deque[int]) -> int:
    if not ids:
        return 0
    counts: Dict[int, int] = {}
    for val in ids:
        counts[val] = counts.get(val, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _mean_probs(probs: Deque[np.ndarray]) -> np.ndarray:
    if not probs:
        return np.array([])
    stack = np.stack(list(probs), axis=0)
    return np.mean(stack, axis=0)


def _adjacent_to(last_finger: int, candidate: int) -> bool:
    chain = [1, 2, 3, 4, 5]
    if last_finger not in chain or candidate not in chain:
        return False
    return abs(chain.index(last_finger) - chain.index(candidate)) == 1


def postprocess_predictions(
    action_probs: np.ndarray,
    finger_probs: np.ndarray,
    settings: PostprocessSettings,
    state: PostprocessState,
) -> Dict[str, object]:
    smoothing_window = max(1, int(settings.smoothing_window))

    raw_action_id = int(np.argmax(action_probs))
    raw_finger_id = int(np.argmax(finger_probs))

    smoothed_action_probs = action_probs
    smoothed_finger_probs = finger_probs
    smoothed_action_id = raw_action_id
    smoothed_finger_id = raw_finger_id

    if settings.smoothing_enabled:
        if settings.smoothing_method == "ema":
            alpha = 2.0 / (smoothing_window + 1.0)
            if state.ema_action is None:
                state.ema_action = action_probs.copy()
            else:
                state.ema_action = alpha * action_probs + (1 - alpha) * state.ema_action
            if state.ema_finger is None:
                state.ema_finger = finger_probs.copy()
            else:
                state.ema_finger = alpha * finger_probs + (1 - alpha) * state.ema_finger
            smoothed_action_probs = state.ema_action
            smoothed_finger_probs = state.ema_finger
            smoothed_action_id = int(np.argmax(smoothed_action_probs))
            smoothed_finger_id = int(np.argmax(smoothed_finger_probs))
        else:
            state.action_ids.append(raw_action_id)
            state.finger_ids.append(raw_finger_id)
            state.action_probs.append(action_probs)
            state.finger_probs.append(finger_probs)
            while len(state.action_ids) > smoothing_window:
                state.action_ids.popleft()
            while len(state.finger_ids) > smoothing_window:
                state.finger_ids.popleft()
            while len(state.action_probs) > smoothing_window:
                state.action_probs.popleft()
            while len(state.finger_probs) > smoothing_window:
                state.finger_probs.popleft()
            smoothed_action_id = _vote_majority(state.action_ids)
            smoothed_finger_id = _vote_majority(state.finger_ids)
            smoothed_action_probs = _mean_probs(state.action_probs)
            smoothed_finger_probs = _mean_probs(state.finger_probs)

    action_conf = float(np.max(smoothed_action_probs)) if smoothed_action_probs.size else 0.0
    finger_conf = float(np.max(smoothed_finger_probs)) if smoothed_finger_probs.size else 0.0

    decision_reason = "vote_commit" if settings.smoothing_method == "vote" else "ema_commit"

    if action_conf < float(settings.threshold_action):
        committed_action = 0
        committed_finger = 0
        state.pending_action = None
        state.pending_count = 0
        decision_reason = "below_threshold"
    else:
        candidate_action = int(smoothed_action_id)
        committed_action = candidate_action
        if settings.hysteresis_enabled and candidate_action in {1, 2} and state.last_action in {1, 2}:
            if candidate_action != state.last_action:
                if action_conf < float(settings.threshold_action) + float(settings.hysteresis_margin):
                    committed_action = state.last_action
                    decision_reason = "hysteresis_hold"
                else:
                    if state.pending_action == candidate_action:
                        state.pending_count += 1
                    else:
                        state.pending_action = candidate_action
                        state.pending_count = 1
                    if state.pending_count >= int(settings.hysteresis_frames):
                        committed_action = candidate_action
                        state.pending_action = None
                        state.pending_count = 0
                        decision_reason = "hysteresis_commit"
                    else:
                        committed_action = state.last_action
                        decision_reason = "hysteresis_hold"
            else:
                state.pending_action = None
                state.pending_count = 0

        committed_finger = int(smoothed_finger_id)
        if committed_action == 0:
            committed_finger = 0
        else:
            if settings.adjacency_enabled and smoothed_finger_probs.size:
                sorted_idx = np.argsort(smoothed_finger_probs)[::-1]
                top1, top2 = int(sorted_idx[0]), int(sorted_idx[1]) if len(sorted_idx) > 1 else int(sorted_idx[0])
                if top1 != top2:
                    gap = float(smoothed_finger_probs[top1] - smoothed_finger_probs[top2])
                    if gap < float(settings.finger_delta) and state.last_finger != 0 and _adjacent_to(state.last_finger, top2):
                        committed_finger = top2
                        decision_reason = "adjacent_correction"
            if finger_conf < float(settings.threshold_finger):
                committed_finger = 0
                if decision_reason not in {"below_threshold", "hysteresis_hold"}:
                    decision_reason = "finger_below_threshold"

    if committed_action == state.last_action:
        state.frames_in_state += 1
    else:
        state.frames_in_state = 1

    state.last_action = int(committed_action)
    state.last_finger = int(committed_finger)

    return {
        "committed_action_id": int(committed_action),
        "committed_finger_id": int(committed_finger),
        "raw_top_action_id": int(raw_action_id),
        "raw_top_finger_id": int(raw_finger_id),
        "action_conf": float(action_conf),
        "finger_conf": float(finger_conf),
        "smoothed_action_id": int(smoothed_action_id),
        "smoothed_finger_id": int(smoothed_finger_id),
        "decision_reason": decision_reason,
        "frames_in_state": int(state.frames_in_state),
    }
