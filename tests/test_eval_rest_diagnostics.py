import importlib.util
from pathlib import Path

import numpy as np


def _load_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "3_evaluate_model.py"
    spec = importlib.util.spec_from_file_location("step3_evaluate_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rest_event_breakdown_flags_relabel_and_prune_cases():
    mod = _load_eval_module()
    y_action_true = np.array([0] * 10, dtype=np.int64)
    y_finger_true = np.array([0] * 10, dtype=np.int64)

    action_probs = np.array(
        [
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.10, 0.45, 0.45],
            [0.10, 0.10, 0.80],
            [0.10, 0.45, 0.45],
        ],
        dtype=np.float32,
    )
    finger_probs = np.array(
        [
            [0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
            [0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
            [0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
            [0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
            [0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
            [0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
            [0.01, 0.01, 0.0, 0.98, 0.0, 0.0],
            [0.01, 0.01, 0.98, 0.0, 0.0, 0.0],
            [0.01, 0.01, 0.0, 0.98, 0.0, 0.0],
            [0.01, 0.01, 0.98, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    payload = mod._compute_prediction_metrics(
        action_probs=action_probs,
        finger_probs=finger_probs,
        applicability_probs=None,
        y_action_true=y_action_true,
        y_finger_true=y_finger_true,
    )
    meta = {
        "event_id": np.array([1] * 5 + [2] * 5, dtype=np.int64),
        "session_id": np.array(["S1"] * 10, dtype="U"),
        "window_start": np.arange(10, dtype=np.float32),
        "window_end": np.arange(10, dtype=np.float32) + 0.25,
    }
    breakdown, flags = mod._build_rest_event_breakdown(
        indices=np.arange(10, dtype=np.int64),
        y_action_true=y_action_true,
        action_probs=action_probs,
        action_preds=payload["action_preds"],
        finger_preds=payload["finger_preds"],
        meta=meta,
    )

    assert [row["event_id"] for row in breakdown] == [1, 2]
    assert len(flags) == 2
    flag_by_event = {row["event_id"]: row for row in flags}
    assert flag_by_event[1]["recommended_action"] == "relabel"
    assert flag_by_event[1]["dominant_non_rest_pair"] == "OPEN+THUMB"
    assert flag_by_event[2]["recommended_action"] == "prune"
    assert flag_by_event[2]["rest_tpr"] < 0.20


def test_compute_prediction_metrics_reports_rest_and_pair_validity():
    mod = _load_eval_module()
    action_probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )
    finger_probs = np.array(
        [
            [0.9, 0.05, 0.05, 0.0, 0.0, 0.0],
            [0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
            [0.01, 0.01, 0.98, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    payload = mod._compute_prediction_metrics(
        action_probs=action_probs,
        finger_probs=finger_probs,
        applicability_probs=np.array([0.2, 0.9, 0.8], dtype=np.float32),
        y_action_true=np.array([0, 1, 2], dtype=np.int64),
        y_finger_true=np.array([0, 1, 2], dtype=np.int64),
    )

    assert payload["metrics"]["action_acc"] == 1.0
    assert payload["metrics"]["rest_tpr"] == 1.0
    assert payload["metrics"]["raw_non_rest_none_count"] == 0
    assert payload["metrics"]["raw_rest_non_none_count"] == 0
    assert payload["metrics"]["committed_non_rest_none_count"] == 0
    assert payload["metrics"]["committed_rest_non_none_count"] == 0
    assert payload["metrics"]["sent_non_rest_none_count"] is None
    assert payload["metrics"]["sent_rest_non_none_count"] is None
    assert payload["metrics"]["applicability_fp_rate_on_true_rest"] == 0.0
    assert payload["metrics"]["applicability_fn_rate_on_true_non_rest"] == 0.0
    assert payload["metrics"]["action_applicability_disagreement_rate"] == 0.0
    assert payload["metrics"]["deployment_pair_invariant_ok"] is True
    assert payload["metrics"]["raw_invalid_pair_rate"] == 0.0
    assert payload["pair_counts"]["REST+NONE"] == 1
    per_finger = payload["metrics"]["finger_metrics_by_class_non_rest"]
    assert per_finger["THUMB"]["correct"] == 1
    assert per_finger["THUMB"]["support"] == 1
    assert per_finger["THUMB"]["accuracy"] == 1.0
    assert per_finger["INDEX"]["correct"] == 1
    assert per_finger["INDEX"]["support"] == 1
    assert per_finger["INDEX"]["accuracy"] == 1.0


def test_compute_prediction_metrics_respects_applicability_threshold() -> None:
    mod = _load_eval_module()
    payload = mod._compute_prediction_metrics(
        action_probs=np.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
            ],
            dtype=np.float32,
        ),
        finger_probs=np.array(
            [
                [0.9, 0.05, 0.05, 0.0, 0.0, 0.0],
                [0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        applicability_probs=np.array([0.45, 0.55], dtype=np.float32),
        y_action_true=np.array([0, 1], dtype=np.int64),
        y_finger_true=np.array([0, 1], dtype=np.int64),
        threshold_applicability=0.4,
    )

    assert payload["metrics"]["applicability_fp_rate_on_true_rest"] == 1.0
    assert payload["metrics"]["applicability_fn_rate_on_true_non_rest"] == 0.0


def test_compute_prediction_metrics_separates_raw_none_from_committed_decode():
    mod = _load_eval_module()
    action_probs = np.array(
        [
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ],
        dtype=np.float32,
    )
    finger_probs = np.array(
        [
            [0.85, 0.10, 0.03, 0.01, 0.01, 0.00],
            [0.80, 0.05, 0.05, 0.05, 0.03, 0.02],
        ],
        dtype=np.float32,
    )
    payload = mod._compute_prediction_metrics(
        action_probs=action_probs,
        finger_probs=finger_probs,
        applicability_probs=None,
        y_action_true=np.array([1, 2], dtype=np.int64),
        y_finger_true=np.array([1, 1], dtype=np.int64),
    )

    assert payload["metrics"]["raw_non_rest_none_count"] == 2
    assert payload["metrics"]["raw_non_rest_none_rate"] == 1.0
    assert payload["metrics"]["committed_non_rest_none_count"] == 0
    assert payload["metrics"]["committed_non_rest_none_rate"] == 0.0
    assert payload["metrics"]["committed_rest_non_none_count"] == 0
    assert payload["metrics"]["deployment_pair_invariant_ok"] is True
    assert payload["pair_counts"]["OPEN+THUMB"] == 1
    assert payload["pair_counts"]["CLOSE+THUMB"] == 1


def test_primary_benchmark_surfaces_applicability_and_committed_rest_metrics():
    mod = _load_eval_module()
    benchmark = mod._build_primary_benchmark(
        y_action_test=np.array([0, 1, 2], dtype=np.int64),
        y_finger_test=np.array([0, 1, 2], dtype=np.int64),
        metrics={
            "action_acc": 0.9,
            "joint_acc": 0.8,
            "joint_acc_non_rest": 0.75,
            "finger_acc_non_rest": 0.85,
            "finger_metrics_by_class_non_rest": {
                "THUMB": {"accuracy": 1.0, "support": 1, "predicted_count": 1, "correct": 1},
            },
            "rest_tpr": 0.7,
            "rest_precision": 0.6,
            "action_ece": 0.1,
            "applicability_fp_rate_on_true_rest": 0.2,
            "applicability_fn_rate_on_true_non_rest": 0.1,
            "action_applicability_disagreement_rate": 0.05,
            "raw_non_rest_none_rate": 0.0,
            "raw_rest_non_none_rate": 0.3,
            "committed_rest_non_none_rate": 0.0,
            "committed_non_rest_none_rate": 0.0,
            "deployment_pair_invariant_ok": True,
        },
    )

    metrics = benchmark["metrics"]
    assert metrics["applicability_fp_rate_on_true_rest"] == 0.2
    assert metrics["applicability_fn_rate_on_true_non_rest"] == 0.1
    assert metrics["action_applicability_disagreement_rate"] == 0.05
    assert metrics["committed_rest_non_none_rate"] == 0.0
    assert metrics["finger_metrics_by_class_non_rest"]["THUMB"]["accuracy"] == 1.0
