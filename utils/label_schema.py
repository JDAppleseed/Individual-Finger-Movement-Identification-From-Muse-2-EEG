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
ACTIVE_FINGER_IDS = (
    FINGER_THUMB,
    FINGER_INDEX,
    FINGER_MIDDLE,
    FINGER_RING,
    FINGER_PINKY,
)

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
    return finger_id in {
        FINGER_THUMB,
        FINGER_INDEX,
        FINGER_MIDDLE,
        FINGER_RING,
        FINGER_PINKY,
    }


def uses_active_finger_head(n_fingers: int) -> bool:
    try:
        return int(n_fingers) == len(ACTIVE_FINGER_IDS)
    except Exception:
        return False


def model_index_to_finger_id(model_index: int, n_fingers: int) -> int:
    model_index = int(model_index)
    if uses_active_finger_head(n_fingers):
        if model_index < 0 or model_index >= len(ACTIVE_FINGER_IDS):
            raise ValueError(
                f"model_index {model_index} out of range for active finger head with {n_fingers} classes"
            )
        return int(ACTIVE_FINGER_IDS[model_index])
    return model_index


def finger_id_to_model_index(finger_id: int, n_fingers: int) -> int:
    finger_id = int(finger_id)
    if uses_active_finger_head(n_fingers):
        if finger_id == FINGER_NONE:
            raise ValueError("FINGER_NONE is not represented in an active finger head")
        try:
            return int(ACTIVE_FINGER_IDS.index(finger_id))
        except ValueError as exc:
            raise ValueError(
                f"finger_id {finger_id} is not a valid active finger label"
            ) from exc
    return finger_id


def decode_finger_prediction(finger_scores) -> int:
    finger_arr = np.asarray(finger_scores)
    if finger_arr.size == 0:
        return int(FINGER_NONE)
    raw_index = int(np.argmax(finger_arr))
    return model_index_to_finger_id(raw_index, int(finger_arr.shape[-1]))


def decode_finger_predictions(finger_scores) -> np.ndarray:
    finger_arr = np.asarray(finger_scores)
    if finger_arr.ndim != 2:
        raise ValueError(
            f"finger_scores must be 2-D for batch decode, got shape {finger_arr.shape}"
        )
    raw_idx = np.argmax(finger_arr, axis=1).astype(np.int64)
    if uses_active_finger_head(int(finger_arr.shape[1])):
        lookup = np.asarray(ACTIVE_FINGER_IDS, dtype=np.int64)
        return lookup[raw_idx]
    return raw_idx


def decode_active_finger_prediction(finger_scores) -> int:
    finger_arr = np.asarray(finger_scores)
    if finger_arr.size == 0:
        return int(FINGER_NONE)
    n_fingers = int(finger_arr.shape[-1])
    if uses_active_finger_head(n_fingers):
        return decode_finger_prediction(finger_arr)
    if n_fingers <= 1:
        return int(FINGER_NONE)
    raw_index = int(np.argmax(finger_arr[1:])) + 1
    return model_index_to_finger_id(raw_index, n_fingers)


def decode_active_finger_predictions(finger_scores) -> np.ndarray:
    finger_arr = np.asarray(finger_scores)
    if finger_arr.ndim != 2:
        raise ValueError(
            f"finger_scores must be 2-D for batch decode, got shape {finger_arr.shape}"
        )
    n_fingers = int(finger_arr.shape[1])
    if uses_active_finger_head(n_fingers):
        return decode_finger_predictions(finger_arr)
    if n_fingers <= 1:
        return np.zeros(finger_arr.shape[0], dtype=np.int64)
    raw_idx = np.argmax(finger_arr[:, 1:], axis=1).astype(np.int64) + 1
    return raw_idx


def decode_finger_prediction_for_action(action_id: int, finger_scores) -> int:
    if int(action_id) == ACTION_REST:
        return int(FINGER_NONE)
    return decode_active_finger_prediction(finger_scores)


def decode_finger_predictions_for_actions(action_ids, finger_scores) -> np.ndarray:
    action_arr = np.asarray(action_ids, dtype=np.int64).reshape(-1)
    finger_arr = np.asarray(finger_scores)
    if finger_arr.ndim != 2:
        raise ValueError(
            f"finger_scores must be 2-D for batch decode, got shape {finger_arr.shape}"
        )
    if finger_arr.shape[0] != action_arr.shape[0]:
        raise ValueError(
            f"finger_scores rows {finger_arr.shape[0]} do not match action_ids length {action_arr.shape[0]}"
        )
    out = decode_active_finger_predictions(finger_arr)
    out = np.asarray(out, dtype=np.int64).reshape(-1)
    out[action_arr == ACTION_REST] = int(FINGER_NONE)
    return out


def prediction_pair_diagnostics(
    action_ids,
    raw_finger_ids,
    *,
    committed_finger_ids=None,
):
    action_arr = np.asarray(action_ids, dtype=np.int64).reshape(-1)
    raw_finger_arr = np.asarray(raw_finger_ids, dtype=np.int64).reshape(-1)
    if action_arr.shape != raw_finger_arr.shape:
        raise ValueError(
            f"action_ids and raw_finger_ids shape mismatch: {action_arr.shape} vs {raw_finger_arr.shape}"
        )

    raw_non_rest_none = (action_arr != ACTION_REST) & (raw_finger_arr == int(FINGER_NONE))
    raw_rest_non_none = (action_arr == ACTION_REST) & (raw_finger_arr != int(FINGER_NONE))

    result = {
        "raw_non_rest_none_count": int(np.sum(raw_non_rest_none)),
        "raw_non_rest_none_rate": (
            float(np.mean(raw_non_rest_none)) if raw_non_rest_none.size else None
        ),
        "raw_rest_non_none_count": int(np.sum(raw_rest_non_none)),
        "raw_rest_non_none_rate": (
            float(np.mean(raw_rest_non_none)) if raw_rest_non_none.size else None
        ),
    }

    if committed_finger_ids is not None:
        committed_finger_arr = np.asarray(committed_finger_ids, dtype=np.int64).reshape(-1)
        if action_arr.shape != committed_finger_arr.shape:
            raise ValueError(
                f"action_ids and committed_finger_ids shape mismatch: {action_arr.shape} vs {committed_finger_arr.shape}"
            )
        committed_non_rest_none = (action_arr != ACTION_REST) & (
            committed_finger_arr == int(FINGER_NONE)
        )
        result["committed_non_rest_none_count"] = int(np.sum(committed_non_rest_none))
        result["committed_non_rest_none_rate"] = (
            float(np.mean(committed_non_rest_none))
            if committed_non_rest_none.size
            else None
        )
        result["deployment_pair_invariant_ok"] = bool(
            result["committed_non_rest_none_count"] == 0
        )

    return result


def finger_confidence_for_id(finger_scores, finger_id: int) -> float:
    finger_arr = np.asarray(finger_scores)
    if finger_arr.size == 0:
        return 0.0
    try:
        model_idx = finger_id_to_model_index(int(finger_id), int(finger_arr.shape[-1]))
    except ValueError:
        return 0.0
    if model_idx < 0 or model_idx >= int(finger_arr.shape[-1]):
        return 0.0
    return float(finger_arr[model_idx])


def finger_confidences_for_ids(finger_scores, finger_ids) -> np.ndarray:
    finger_arr = np.asarray(finger_scores)
    finger_ids_arr = np.asarray(finger_ids, dtype=np.int64).reshape(-1)
    if finger_arr.ndim != 2:
        raise ValueError(
            f"finger_scores must be 2-D for batch confidence lookup, got shape {finger_arr.shape}"
        )
    if finger_arr.shape[0] != finger_ids_arr.shape[0]:
        raise ValueError(
            f"finger_scores rows {finger_arr.shape[0]} do not match finger_ids length {finger_ids_arr.shape[0]}"
        )
    out = np.zeros(finger_ids_arr.shape[0], dtype=finger_arr.dtype)
    n_fingers = int(finger_arr.shape[1])
    if uses_active_finger_head(n_fingers):
        for model_idx, finger_id in enumerate(ACTIVE_FINGER_IDS):
            mask = finger_ids_arr == int(finger_id)
            if np.any(mask):
                out[mask] = finger_arr[mask, model_idx]
        return out
    mask = (finger_ids_arr >= 0) & (finger_ids_arr < n_fingers)
    if np.any(mask):
        rows = np.nonzero(mask)[0]
        out[mask] = finger_arr[rows, finger_ids_arr[mask]]
    return out


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
    action_id = int(np.argmax(action_arr)) if action_arr.size else int(ACTION_REST)
    finger_id = decode_finger_prediction_for_action(action_id, finger_scores)
    return action_id, int(finger_id)
