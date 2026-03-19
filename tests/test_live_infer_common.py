from __future__ import annotations

import numpy as np

from utils.live_infer_common import (
    ActuationDecision,
    compute_replay_metrics,
    debounced_should_send,
    resolve_actuation_candidate,
)


def test_debounced_should_send_respects_cooldown_and_repeat() -> None:
    decision = ActuationDecision(finger_id=1, action_id=2, prob=0.9)

    assert debounced_should_send(
        decision,
        last_sent=None,
        stable_count=1,
        required_stability=1,
        last_send_time_ms=None,
        current_time_ms=1_000.0,
        cooldown_ms=250,
        repeat_same_ms=500,
    )

    assert not debounced_should_send(
        decision,
        last_sent=(1, 2),
        stable_count=1,
        required_stability=1,
        last_send_time_ms=900.0,
        current_time_ms=1_200.0,
        cooldown_ms=250,
        repeat_same_ms=500,
    )

    assert debounced_should_send(
        decision,
        last_sent=(1, 2),
        stable_count=1,
        required_stability=1,
        last_send_time_ms=600.0,
        current_time_ms=1_200.0,
        cooldown_ms=250,
        repeat_same_ms=500,
    )

    assert not debounced_should_send(
        decision,
        last_sent=(3, 1),
        stable_count=1,
        required_stability=1,
        last_send_time_ms=900.0,
        current_time_ms=1_000.0,
        cooldown_ms=250,
        repeat_same_ms=500,
    )


def test_resolve_actuation_candidate_requires_stable_nonzero_finger() -> None:
    unstable = [
        ActuationDecision(finger_id=3, action_id=1, prob=0.7),
        ActuationDecision(finger_id=0, action_id=1, prob=0.6),
        ActuationDecision(finger_id=3, action_id=2, prob=0.8),
    ]
    unstable_result = resolve_actuation_candidate(
        unstable,
        required_finger_stability=3,
    )
    assert unstable_result["reason"] == "finger_stability"
    assert unstable_result["decision"].finger_id == 0
    assert unstable_result["decision"].action_id == 0

    stable = [
        ActuationDecision(finger_id=3, action_id=1, prob=0.7),
        ActuationDecision(finger_id=3, action_id=1, prob=0.8),
        ActuationDecision(finger_id=3, action_id=2, prob=0.9),
    ]
    stable_result = resolve_actuation_candidate(
        stable,
        required_finger_stability=3,
    )
    assert stable_result["reason"] == "finger_majority_action_vote"
    assert stable_result["decision"].finger_id == 3
    assert stable_result["decision"].action_id == 1
    assert stable_result["resolved_finger_id"] == 3


def test_compute_replay_metrics_reports_active_finger_behavior() -> None:
    records = [
        {
            "committed_action_id": 1,
            "committed_finger_id": 2,
            "committed_pair_valid": True,
            "applicability_gate_ok": True,
            "finger_applicable_prob": 0.95,
            "actuation_sent": True,
            "actuation_target_action_id": 1,
            "actuation_target_finger_id": 2,
            "offline_compute_ms": 5.0,
        },
        {
            "committed_action_id": 1,
            "committed_finger_id": 2,
            "committed_pair_valid": True,
            "applicability_gate_ok": True,
            "finger_applicable_prob": 0.90,
            "actuation_sent": True,
            "actuation_target_action_id": 1,
            "actuation_target_finger_id": 2,
            "offline_compute_ms": 5.0,
        },
        {
            "committed_action_id": 0,
            "committed_finger_id": 0,
            "committed_pair_valid": True,
            "applicability_gate_ok": False,
            "finger_applicable_prob": 0.10,
            "actuation_sent": False,
            "actuation_target_action_id": 0,
            "actuation_target_finger_id": 0,
            "offline_compute_ms": 5.0,
        },
    ]
    y_action_true = np.asarray([1, 1, 0], dtype=np.int64)
    y_finger_true = np.asarray([2, 2, 0], dtype=np.int64)
    window_start_s = np.asarray([0.0, 0.05, 0.10], dtype=np.float32)
    window_end_s = np.asarray([0.25, 0.30, 0.35], dtype=np.float32)
    trial_ids = np.asarray([1, 1, 1], dtype=np.int64)
    session_ids = np.asarray(["S1", "S1", "S1"], dtype="U")
    event_ids = np.asarray([7, 7, 8], dtype=np.int64)
    event_onset_s = np.asarray([0.0, 0.0, 0.10], dtype=np.float32)

    metrics = compute_replay_metrics(
        records=records,
        y_action_true=y_action_true,
        y_finger_true=y_finger_true,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        trial_ids=trial_ids,
        session_ids=session_ids,
        event_ids=event_ids,
        event_onset_s=event_onset_s,
    )

    assert metrics["committed_action_acc"] == 1.0
    assert metrics["committed_joint_acc"] == 1.0
    assert metrics["committed_finger_acc_non_rest"] == 1.0
    assert metrics["would_send_window_precision_non_rest"] == 1.0
    assert metrics["would_send_window_recall_non_rest"] == 1.0
    assert metrics["false_actuation_rate_rest"] == 0.0
    assert metrics["non_rest_none_count"] == 0
    assert metrics["committed_non_rest_none_count"] == 0
    assert metrics["committed_rest_non_none_count"] == 0
    assert metrics["sent_non_rest_none_count"] == 0
    assert metrics["sent_rest_non_none_count"] == 0
    assert metrics["applicability_fp_rate_on_true_rest"] == 0.0
    assert metrics["applicability_fn_rate_on_true_non_rest"] == 0.0
    assert metrics["action_applicability_disagreement_rate"] == 0.0
    assert metrics["deployment_pair_invariant_ok"] is True
    assert metrics["committed_segment_overlap"]["truth_segment_count"] == 1


def test_compute_replay_metrics_flags_invalid_committed_or_sent_pairs() -> None:
    records = [
        {
            "committed_action_id": 1,
            "committed_finger_id": 0,
            "committed_pair_valid": False,
            "applicability_gate_ok": True,
            "finger_applicable_prob": 0.9,
            "actuation_sent": True,
            "actuation_target_action_id": 1,
            "actuation_target_finger_id": 0,
            "offline_compute_ms": 5.0,
        },
        {
            "committed_action_id": 0,
            "committed_finger_id": 2,
            "committed_pair_valid": False,
            "applicability_gate_ok": True,
            "finger_applicable_prob": 0.8,
            "actuation_sent": True,
            "actuation_target_action_id": 0,
            "actuation_target_finger_id": 2,
            "offline_compute_ms": 5.0,
        }
    ]

    metrics = compute_replay_metrics(
        records=records,
        y_action_true=np.asarray([1, 0], dtype=np.int64),
        y_finger_true=np.asarray([1, 0], dtype=np.int64),
        window_start_s=np.asarray([0.0, 0.25], dtype=np.float32),
        window_end_s=np.asarray([0.25, 0.50], dtype=np.float32),
        trial_ids=np.asarray([1, 1], dtype=np.int64),
        session_ids=np.asarray(["S1", "S1"], dtype="U"),
        event_ids=np.asarray([7, 8], dtype=np.int64),
        event_onset_s=np.asarray([0.0, 0.25], dtype=np.float32),
    )

    assert metrics["committed_non_rest_none_count"] == 1
    assert metrics["committed_rest_non_none_count"] == 1
    assert metrics["sent_non_rest_none_count"] == 1
    assert metrics["sent_rest_non_none_count"] == 1
    assert metrics["deployment_pair_invariant_ok"] is False


def test_compute_replay_metrics_uses_probability_not_rest_bypassed_gate_for_applicability() -> None:
    records = [
        {
            "committed_action_id": 0,
            "committed_finger_id": 0,
            "committed_pair_valid": True,
            "applicability_gate_ok": True,
            "finger_applicable_prob": 0.10,
            "actuation_sent": False,
            "actuation_target_action_id": 0,
            "actuation_target_finger_id": 0,
            "offline_compute_ms": 5.0,
        },
        {
            "committed_action_id": 1,
            "committed_finger_id": 2,
            "committed_pair_valid": True,
            "applicability_gate_ok": False,
            "finger_applicable_prob": 0.90,
            "actuation_sent": False,
            "actuation_target_action_id": 0,
            "actuation_target_finger_id": 0,
            "offline_compute_ms": 5.0,
        },
    ]

    metrics = compute_replay_metrics(
        records=records,
        y_action_true=np.asarray([0, 1], dtype=np.int64),
        y_finger_true=np.asarray([0, 2], dtype=np.int64),
        window_start_s=np.asarray([0.0, 0.25], dtype=np.float32),
        window_end_s=np.asarray([0.25, 0.50], dtype=np.float32),
        trial_ids=np.asarray([1, 1], dtype=np.int64),
        session_ids=np.asarray(["S1", "S1"], dtype="U"),
        event_ids=np.asarray([8, 9], dtype=np.int64),
        event_onset_s=np.asarray([0.0, 0.25], dtype=np.float32),
        applicability_threshold=0.5,
    )

    assert metrics["applicability_fp_rate_on_true_rest"] == 0.0
    assert metrics["applicability_fn_rate_on_true_non_rest"] == 0.0
    assert metrics["action_applicability_disagreement_rate"] == 0.0
