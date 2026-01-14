import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from demo_backend.postprocess import (
        PostprocessSettings,
        PostprocessState,
        postprocess_predictions,
    )

    settings = PostprocessSettings(
        smoothing_enabled=True,
        smoothing_method="vote",
        smoothing_window=5,
        hysteresis_enabled=True,
        hysteresis_frames=3,
        threshold_action=0.6,
        threshold_finger=0.6,
        adjacency_enabled=False,
    )
    state = PostprocessState()

    # Rapid flips between OPEN(1) and CLOSE(2); hysteresis should stabilize.
    seq = [1, 2, 1, 2, 1, 2, 1, 1, 1]
    committed = []
    for action_id in seq:
        action_probs = np.array([0.05, 0.475, 0.475])
        action_probs[action_id] = 0.8
        action_probs = action_probs / action_probs.sum()
        finger_probs = np.array([0.7, 0.1, 0.1, 0.1, 0.0, 0.0])
        post = postprocess_predictions(action_probs, finger_probs, settings, state)
        committed.append(int(post["committed_action_id"]))

    if len(set(committed[-3:])) != 1:
        raise SystemExit("Postprocess hysteresis did not stabilize output")

    # Below-threshold forces REST
    low_action_probs = np.array([0.34, 0.33, 0.33])
    low_finger_probs = np.array([0.34, 0.33, 0.33, 0.0, 0.0, 0.0])
    post_low = postprocess_predictions(
        low_action_probs, low_finger_probs, settings, state
    )
    if int(post_low["committed_action_id"]) != 0:
        raise SystemExit("Postprocess did not force REST below threshold")

    print("✅ self_check_postprocess passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
