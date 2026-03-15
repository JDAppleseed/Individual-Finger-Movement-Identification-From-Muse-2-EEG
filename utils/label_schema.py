"""
Shared label schema for action/finger classification.
"""

from typing import Optional, Tuple

import numpy as np

ACTION_REST = 0
ACTION_OPEN = 1
ACTION_CLOSE = 2

FINGER_NONE = 0
FINGER_THUMB = 1
FINGER_INDEX = 2
FINGER_MIDDLE = 3
FINGER_RING = 4
FINGER_PINKY = 5

ACTION_NAMES = {
    ACTION_REST: "REST",
    ACTION_OPEN: "OPEN",
    ACTION_CLOSE: "CLOSE",
}

FINGER_NAMES = {
    FINGER_NONE: "NONE",
    FINGER_THUMB: "THUMB",
    FINGER_INDEX: "INDEX",
    FINGER_MIDDLE: "MIDDLE",
    FINGER_RING: "RING",
    FINGER_PINKY: "PINKY",
}


def is_valid_action_finger(action_id: int, finger_id: int) -> bool:
    if action_id == ACTION_REST:
        return finger_id == FINGER_NONE
    if finger_id == FINGER_NONE:
        return action_id in {ACTION_OPEN, ACTION_CLOSE}
    return finger_id in {
        FINGER_THUMB,
        FINGER_INDEX,
        FINGER_MIDDLE,
        FINGER_RING,
        FINGER_PINKY,
    }


def event_type_for(
    action_id: int, finger_id: int, override: Optional[str] = None
) -> str:
    if override:
        return override
    if action_id == ACTION_REST:
        return "rest"
    finger_name = FINGER_NAMES.get(finger_id, "unknown").lower()
    action_name = ACTION_NAMES.get(action_id, "action").lower()
    return f"{finger_name}_{action_name}"


def enforce_prediction_pair(action_id: int, finger_id: int) -> Tuple[int, int]:
    action_id = int(action_id)
    finger_id = int(finger_id)
    if action_id == ACTION_REST:
        return action_id, int(FINGER_NONE)
    return action_id, finger_id


def enforce_prediction_pairs(action_ids, finger_ids):
    action_arr = np.asarray(action_ids, dtype=np.int64).reshape(-1)
    finger_arr = np.asarray(finger_ids, dtype=np.int64).reshape(-1).copy()
    if action_arr.shape != finger_arr.shape:
        raise ValueError(
            f"action_ids and finger_ids shape mismatch: {action_arr.shape} vs {finger_arr.shape}"
        )
    finger_arr[action_arr == ACTION_REST] = int(FINGER_NONE)
    return action_arr, finger_arr


def decode_prediction_pair(action_scores, finger_scores) -> Tuple[int, int]:
    action_arr = np.asarray(action_scores)
    finger_arr = np.asarray(finger_scores)
    action_id = int(np.argmax(action_arr)) if action_arr.size else int(ACTION_REST)
    finger_id = int(np.argmax(finger_arr)) if finger_arr.size else int(FINGER_NONE)
    return enforce_prediction_pair(action_id, finger_id)
