import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_train_module():
    module_path = Path(__file__).resolve().parents[1] / "2_train_model.py"
    spec = importlib.util.spec_from_file_location("step2_train_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_session_equalized_rest_weights_balance_rest_mass():
    mod = _load_train_module()
    y_action = np.array([0] * 8 + [0] * 2 + [1] * 6, dtype=np.int64)
    meta = {
        "session_id": np.array(
            ["rest_heavy"] * 8 + ["rest_light"] * 2 + ["rest_heavy"] * 6,
            dtype="U",
        )
    }

    weights, summary = mod._build_train_sample_weights(
        y_action,
        meta,
        balance_mode="session_equalized",
    )

    assert weights is not None
    assert summary["enabled"] is True
    rest_heavy_mass = float(np.sum(weights[(y_action == 0) & (meta["session_id"] == "rest_heavy")]))
    rest_light_mass = float(np.sum(weights[(y_action == 0) & (meta["session_id"] == "rest_light")]))
    assert np.isclose(rest_heavy_mass, rest_light_mass)
    assert np.allclose(weights[y_action != 0], 1.0)


def test_session_equalized_rest_weights_disable_without_multiple_rest_sessions():
    mod = _load_train_module()
    y_action = np.array([0, 0, 0, 1, 2], dtype=np.int64)
    meta = {"session_id": np.array(["only"] * len(y_action), dtype="U")}

    weights, summary = mod._build_train_sample_weights(
        y_action,
        meta,
        balance_mode="session_equalized",
    )

    assert weights is None
    assert summary["enabled"] is False
    assert summary["reason"] == "single_rest_session"


def test_auxiliary_rest_session_policy_moves_rest_only_session_to_train_only():
    mod = _load_train_module()
    y_action = np.array([0, 1, 2, 1, 2, 0, 0, 0], dtype=np.int64)
    meta = {
        "session_id": np.array(
            ["move_a"] * 3 + ["move_b"] * 2 + ["rest_only"] * 3,
            dtype="U",
        )
    }

    summary = mod._resolve_auxiliary_rest_sessions(
        y_action,
        meta,
        policy="auto_train_only",
    )

    assert summary["enabled"] is True
    assert summary["aux_sessions"] == ["rest_only"]
    assert sorted(summary["core_sessions"]) == ["move_a", "move_b"]
    assert np.array_equal(summary["aux_idx"], np.array([5, 6, 7], dtype=np.int64))
    assert np.array_equal(summary["core_idx"], np.array([0, 1, 2, 3, 4], dtype=np.int64))


def test_auxiliary_rest_session_policy_noops_when_no_rest_only_session():
    mod = _load_train_module()
    y_action = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    meta = {
        "session_id": np.array(
            ["sess_a"] * 3 + ["sess_b"] * 3,
            dtype="U",
        )
    }

    summary = mod._resolve_auxiliary_rest_sessions(
        y_action,
        meta,
        policy="auto_train_only",
    )

    assert summary["enabled"] is False
    assert summary["reason"] == "no_rest_only_sessions"
    assert np.array_equal(summary["core_idx"], np.arange(len(y_action), dtype=np.int64))
    assert summary["aux_sessions"] == []


def test_core_event_equalized_balances_core_events_and_aux_session_mass():
    mod = _load_train_module()
    y_action = np.array(
        [0] * 6 + [0] * 2 + [1] * 4 + [0] * 10,
        dtype=np.int64,
    )
    meta = {
        "session_id": np.array(
            ["move"] * 12 + ["rest_only"] * 10,
            dtype="U",
        ),
        "event_id": np.array(
            [10] * 6 + [11] * 2 + [20] * 4 + [30] * 10,
            dtype=np.int64,
        ),
    }

    weights, summary = mod._build_train_sample_weights(
        y_action,
        meta,
        balance_mode="core_event_equalized",
    )

    assert weights is not None
    assert summary["enabled"] is True
    core_event_10_mass = float(np.sum(weights[(y_action == 0) & (meta["event_id"] == 10)]))
    core_event_11_mass = float(np.sum(weights[(y_action == 0) & (meta["event_id"] == 11)]))
    aux_mass = float(np.sum(weights[(y_action == 0) & (meta["session_id"] == "rest_only")]))
    assert np.isclose(core_event_10_mass, core_event_11_mass)
    assert np.isclose(core_event_10_mass + core_event_11_mass, 0.70 * 18.0)
    assert np.isclose(aux_mass, 0.30 * 18.0)
    assert np.allclose(weights[y_action != 0], 1.0)


def test_action_weights_override_rest_weight():
    mod = _load_train_module()
    weights, override = mod._resolve_action_class_weights(
        action_weights="1.25,0.9,1.0",
        n_actions=3,
        rest_weight=5.0,
    )

    assert override is True
    assert torch.allclose(weights, torch.tensor([1.25, 0.9, 1.0], dtype=torch.float32))


def test_rest_finger_loss_only_applies_when_enabled():
    mod = _load_train_module()
    finger_logits = torch.tensor([[0.1, 1.2, -0.4]], dtype=torch.float32)
    action_logits = torch.tensor([[2.0, -1.0, -1.5]], dtype=torch.float32)
    y_finger = torch.tensor([1], dtype=torch.long)
    y_action = torch.tensor([0], dtype=torch.long)
    loss_f = torch.nn.CrossEntropyLoss()
    loss_a = torch.nn.CrossEntropyLoss()

    loss0, _, _, rest0 = mod._compute_batch_losses(
        finger_logits=finger_logits,
        action_logits=action_logits,
        y_finger=y_finger,
        y_action=y_action,
        action_loss_fn=loss_a,
        finger_loss_fn=loss_f,
        loss_action_weight=1.0,
        rest_finger_loss_weight=0.0,
        n_finger_classes=3,
    )
    loss1, _, _, rest1 = mod._compute_batch_losses(
        finger_logits=finger_logits,
        action_logits=action_logits,
        y_finger=y_finger,
        y_action=y_action,
        action_loss_fn=loss_a,
        finger_loss_fn=loss_f,
        loss_action_weight=1.0,
        rest_finger_loss_weight=0.5,
        n_finger_classes=3,
    )

    assert float(rest0.item()) == 0.0
    assert float(rest1.item()) > 0.0
    assert float(loss1.item()) > float(loss0.item())


def test_active_finger_head_ignores_rest_finger_loss_and_reindexes_targets():
    mod = _load_train_module()
    finger_logits = torch.tensor([[0.1, 1.2, -0.4, -0.5, -0.6]], dtype=torch.float32)
    action_logits = torch.tensor([[0.2, 1.8, -1.0]], dtype=torch.float32)
    y_finger = torch.tensor([2], dtype=torch.long)
    y_action = torch.tensor([1], dtype=torch.long)
    loss_f = torch.nn.CrossEntropyLoss()
    loss_a = torch.nn.CrossEntropyLoss()

    loss, _, finger_non_rest, finger_rest = mod._compute_batch_losses(
        finger_logits=finger_logits,
        action_logits=action_logits,
        y_finger=y_finger,
        y_action=y_action,
        action_loss_fn=loss_a,
        finger_loss_fn=loss_f,
        loss_action_weight=1.0,
        rest_finger_loss_weight=0.5,
        n_finger_classes=5,
    )

    assert float(finger_non_rest.item()) > 0.0
    assert float(finger_rest.item()) == 0.0
    assert float(loss.item()) > 0.0
