import numpy as np

from utils.event_space_3d import (
    InferenceOutputs,
    assemble_prediction_frame,
    build_base_frame,
    build_plot_figure,
    reduce_to_3d,
)
from utils.live_infer_common import ReplayRuntimeConfig
from utils.postprocess import PostprocessSettings


def test_reduce_to_3d_pca_returns_three_columns():
    representation = np.arange(20, dtype=np.float32).reshape(5, 4)
    coords = reduce_to_3d(representation, reducer="pca", seed=43)
    assert coords.shape == (5, 3)
    assert np.all(np.isfinite(coords))


def test_assemble_prediction_frame_preserves_deployment_semantics_for_active_finger_head():
    y_action = np.array([0, 1, 1], dtype=np.int64)
    y_finger = np.array([0, 2, 1], dtype=np.int64)
    meta = {
        "subject_id": np.array(["2-M16", "2-M16", "2-M16"]),
        "session_id": np.array(["session_a", "session_a", "session_a"]),
        "trial_id": np.array([7, 7, 7], dtype=np.int64),
        "event_id": np.array([11, 11, 11], dtype=np.int64),
        "event_index": np.array([0, 0, 0], dtype=np.int64),
        "window_start": np.array([0.00, 0.05, 0.10], dtype=np.float32),
        "window_end": np.array([0.25, 0.30, 0.35], dtype=np.float32),
        "event_onset_s": np.array([0.00, 0.00, 0.00], dtype=np.float32),
        "event_duration_s": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "assigned_event_type": np.array(["rest", "index_open", "thumb_open"]),
    }
    base = build_base_frame(
        y_action=y_action,
        y_finger=y_finger,
        meta=meta,
        split_labels=np.array(["test", "test", "test"]),
    )

    action_probs = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.05, 0.90, 0.05],
            [0.05, 0.88, 0.07],
        ],
        dtype=np.float32,
    )
    finger_probs = np.array(
        [
            [0.05, 0.70, 0.10, 0.10, 0.05],
            [0.10, 0.75, 0.05, 0.05, 0.05],
            [0.35, 0.25, 0.20, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    applicability_probs = np.array([0.10, 0.90, 0.20], dtype=np.float32)
    outputs = InferenceOutputs(
        representation=np.zeros((3, 4), dtype=np.float32),
        action_logits=np.zeros((3, 3), dtype=np.float32),
        finger_logits=np.zeros((3, 5), dtype=np.float32),
        applicability_logits=np.zeros(3, dtype=np.float32),
        action_probs=action_probs,
        finger_probs=finger_probs,
        applicability_probs=applicability_probs,
    )
    settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.0,
        threshold_finger=0.40,
        threshold_applicability=0.40,
        adjacency_enabled=False,
    )
    runtime = ReplayRuntimeConfig(
        actuation_stability=1,
        actuation_cooldown_ms=0,
        actuation_repeat_ms=0,
        latency_mode="ignore",
    )

    frame = assemble_prediction_frame(
        base_frame=base,
        outputs=outputs,
        postprocess_settings=settings,
        runtime_config=runtime,
    )

    assert frame.loc[0, "raw_top_finger"] == "INDEX"
    assert frame.loc[0, "pred_finger"] == "NONE"
    assert bool(frame.loc[0, "pred_joint_correct"]) is True

    assert frame.loc[1, "pred_action"] == "OPEN"
    assert frame.loc[1, "pred_finger"] == "INDEX"
    assert bool(frame.loc[1, "actuation_sent"]) is True
    assert frame.loc[1, "actuation_suppressed_reason"] == ""

    assert frame.loc[2, "committed_action"] == "OPEN"
    assert frame.loc[2, "committed_finger"] == "THUMB"
    assert bool(frame.loc[2, "finger_gate_ok"]) is False
    assert bool(frame.loc[2, "applicability_gate_ok"]) is False
    assert bool(frame.loc[2, "deployment_pair_valid"]) is True
    assert frame.loc[2, "actuation_suppressed_reason"] == "applicability_gate"
    assert frame.loc[2, "correctness"] == "correct"


def test_build_plot_figure_accepts_correctness_color_mode():
    base = build_base_frame(
        y_action=np.array([0, 1], dtype=np.int64),
        y_finger=np.array([0, 2], dtype=np.int64),
        meta={
            "subject_id": np.array(["2-M16", "2-M16"]),
            "session_id": np.array(["session_a", "session_a"]),
            "trial_id": np.array([1, 1], dtype=np.int64),
            "event_id": np.array([3, 3], dtype=np.int64),
            "event_index": np.array([0, 0], dtype=np.int64),
            "window_start": np.array([0.0, 0.05], dtype=np.float32),
            "window_end": np.array([0.25, 0.30], dtype=np.float32),
        },
        split_labels=np.array(["train", "test"]),
    )
    outputs = InferenceOutputs(
        representation=np.zeros((2, 4), dtype=np.float32),
        action_logits=np.zeros((2, 3), dtype=np.float32),
        finger_logits=np.zeros((2, 5), dtype=np.float32),
        applicability_logits=np.zeros(2, dtype=np.float32),
        action_probs=np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]], dtype=np.float32),
        finger_probs=np.array(
            [[0.8, 0.1, 0.05, 0.03, 0.02], [0.1, 0.7, 0.1, 0.05, 0.05]],
            dtype=np.float32,
        ),
        applicability_probs=np.array([0.1, 0.9], dtype=np.float32),
    )
    frame = assemble_prediction_frame(
        base_frame=base,
        outputs=outputs,
        postprocess_settings=PostprocessSettings(
            smoothing_enabled=False,
            hysteresis_enabled=False,
            threshold_action=0.0,
            threshold_finger=0.0,
            threshold_applicability=0.4,
            adjacency_enabled=False,
        ),
        runtime_config=ReplayRuntimeConfig(
            actuation_stability=1,
            actuation_cooldown_ms=0,
            actuation_repeat_ms=0,
            latency_mode="ignore",
        ),
    )
    frame["emb_x"] = np.array([0.0, 1.0], dtype=np.float32)
    frame["emb_y"] = np.array([0.0, 1.0], dtype=np.float32)
    frame["emb_z"] = np.array([0.0, 1.0], dtype=np.float32)

    fig = build_plot_figure(
        frame,
        color_by="correctness",
        connect_trajectories=True,
        title="correctness plot",
    )
    assert len(fig.data) >= 1
