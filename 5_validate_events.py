"""
STEP 5b — Event Validation & Repair
Flags invalid action/finger combinations and basic timing issues.
Use --apply to fix issues in-place and write edit logs.
"""

import argparse
import pandas as pd

from utils.events_audit import log_event_edit
from utils.label_schema import (
    ACTION_REST,
    FINGER_NONE,
    is_valid_action_finger,
    event_type_for,
)


def validate_events(events):
    issues = []
    for idx, row in events.iterrows():
        onset = float(row["onset_s"])
        duration = float(row["duration_s"])
        action_id = int(row["action_id"])
        finger_id = int(row["finger_id"])

        if onset < 0:
            issues.append((idx, "negative_onset"))
        if duration < 0:
            issues.append((idx, "negative_duration"))
        if not is_valid_action_finger(action_id, finger_id):
            issues.append((idx, "invalid_action_finger"))
    return issues


def repair_event(row):
    before = dict(row)
    action_id = int(row["action_id"])
    finger_id = int(row["finger_id"])

    if action_id == ACTION_REST and finger_id != FINGER_NONE:
        row["finger_id"] = FINGER_NONE
        row["type"] = "rest"
    elif action_id != ACTION_REST and finger_id == FINGER_NONE:
        row["action_id"] = ACTION_REST
        row["finger_id"] = FINGER_NONE
        row["type"] = "rest"
    else:
        row["type"] = event_type_for(int(row["action_id"]), int(row["finger_id"]))

    return before, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply fixes in-place")
    args = parser.parse_args()

    events = pd.read_csv("events.csv")
    issues = validate_events(events)

    if not issues:
        print("✅ No issues found.")
        return

    print("⚠ Issues detected:")
    for idx, issue in issues:
        print(f"- row {idx}: {issue}")

    if not args.apply:
        print("Run with --apply to fix.")
        return

    for idx, issue in issues:
        row = events.loc[idx].copy()
        if issue in {"invalid_action_finger"}:
            before, after = repair_event(row)
            events.loc[idx] = after
            log_event_edit("repair", before, dict(after), note=issue)
        if issue == "negative_onset":
            before = dict(row)
            events.at[idx, "onset_s"] = 0.0
            after = dict(events.loc[idx])
            log_event_edit("repair", before, after, note=issue)
        if issue == "negative_duration":
            before = dict(row)
            events.at[idx, "duration_s"] = 0.0
            after = dict(events.loc[idx])
            log_event_edit("repair", before, after, note=issue)

    events.to_csv("events.csv", index=False)
    print("✅ Applied fixes and saved events.csv")


if __name__ == "__main__":
    main()
