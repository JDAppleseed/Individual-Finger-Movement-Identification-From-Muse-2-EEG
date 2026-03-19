from __future__ import annotations

from typing import Any, Optional, Tuple


def state_dict_has_applicability_head(state_dict: dict[str, Any]) -> bool:
    return bool(
        isinstance(state_dict, dict)
        and "finger_applicability_head.weight" in state_dict
        and "finger_applicability_head.bias" in state_dict
    )


def infer_output_dims_from_state_dict(
    state_dict: dict[str, Any],
) -> Tuple[int, int, bool]:
    return (
        int(state_dict["finger_head.weight"].shape[0]),
        int(state_dict["action_head.weight"].shape[0]),
        bool(state_dict_has_applicability_head(state_dict)),
    )


def unpack_model_outputs(outputs: Any) -> Tuple[Any, Any, Optional[Any]]:
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(
            "Model forward output must be a tuple/list of logits, got "
            f"{type(outputs)!r}."
        )
    if len(outputs) == 2:
        finger_logits, action_logits = outputs
        return finger_logits, action_logits, None
    if len(outputs) == 3:
        finger_logits, action_logits, applicability_logits = outputs
        return finger_logits, action_logits, applicability_logits
    raise ValueError(
        "Model forward output must contain (finger, action) or "
        "(finger, action, applicability) logits."
    )
