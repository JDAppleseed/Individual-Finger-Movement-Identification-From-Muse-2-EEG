#!/usr/bin/env python3
"""
Minimal regression check for none_* action/finger label handling.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_validate_module():
    script_path = REPO_ROOT / "5_validate_events.py"
    spec = spec_from_file_location("validate_events", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    sys.path.insert(0, str(REPO_ROOT))
    from utils.label_schema import (
        ACTION_CLOSE,
        ACTION_OPEN,
        FINGER_NONE,
        event_type_for,
        is_valid_action_finger,
    )

    assert is_valid_action_finger(ACTION_OPEN, FINGER_NONE), (
        "ACTION_OPEN + FINGER_NONE should be valid"
    )
    assert is_valid_action_finger(ACTION_CLOSE, FINGER_NONE), (
        "ACTION_CLOSE + FINGER_NONE should be valid"
    )
    assert event_type_for(ACTION_OPEN, FINGER_NONE) == "none_open"
    assert event_type_for(ACTION_CLOSE, FINGER_NONE) == "none_close"

    validate_mod = load_validate_module()
    repair_event = validate_mod.repair_event

    for action_id in (ACTION_OPEN, ACTION_CLOSE):
        row = pd.Series(
            {
                "action_id": action_id,
                "finger_id": FINGER_NONE,
                "type": event_type_for(action_id, FINGER_NONE),
            }
        )
        _, repaired = repair_event(row)
        assert int(repaired["action_id"]) == action_id
        assert int(repaired["finger_id"]) == FINGER_NONE
        assert repaired["type"] == event_type_for(action_id, FINGER_NONE)

    print("✅ Label schema self-check passed.")


if __name__ == "__main__":
    main()
