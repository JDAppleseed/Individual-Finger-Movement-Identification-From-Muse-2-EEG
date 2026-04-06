import pytest
import numpy as np


def test_replay_viz_import_does_not_require_torch() -> None:
    import visualization.replay_viz as replay_viz

    assert replay_viz.ReplayVisualizer is not None


def test_replay_viz_surfaces_torch_loader_errors(monkeypatch) -> None:
    import visualization.replay_viz as replay_viz

    def _raise() -> None:
        raise RuntimeError("torch missing")

    monkeypatch.setattr(replay_viz, "_load_torch", _raise)
    with pytest.raises(RuntimeError, match="torch missing"):
        replay_viz.ReplayVisualizer("dummy.npz", "model.pt", "scaler.npz")


def test_replay_viz_infers_checkpoint_spec_from_state_dict() -> None:
    import visualization.replay_viz as replay_viz

    spec = replay_viz._infer_replay_checkpoint_spec(
        {
            "finger_head.weight": np.zeros((5, 64), dtype=np.float32),
            "action_head.weight": np.zeros((3, 64), dtype=np.float32),
            "finger_applicability_head.weight": np.zeros((1, 64), dtype=np.float32),
            "finger_applicability_head.bias": np.zeros((1,), dtype=np.float32),
        }
    )

    assert spec.n_fingers == 5
    assert spec.n_actions == 3
    assert spec.has_applicability_head is True


def test_replay_viz_load_state_dict_falls_back_when_weights_only_is_unsupported() -> None:
    import visualization.replay_viz as replay_viz

    class _FakeTorch:
        def __init__(self) -> None:
            self.calls = []

        def load(self, path, **kwargs):
            self.calls.append((path, kwargs))
            if "weights_only" in kwargs:
                raise TypeError("unexpected keyword argument 'weights_only'")
            return {"finger_head.weight": np.zeros((5, 64), dtype=np.float32)}

    fake_torch = _FakeTorch()
    state_dict = replay_viz._load_state_dict(fake_torch, "model.pt")

    assert "finger_head.weight" in state_dict
    assert fake_torch.calls == [
        ("model.pt", {"map_location": "cpu", "weights_only": True}),
        ("model.pt", {"map_location": "cpu"}),
    ]
