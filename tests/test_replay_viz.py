import pytest


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
