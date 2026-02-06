from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_step1_module():
    repo_root = Path(__file__).resolve().parents[1]
    step1_path = repo_root / "1_stream_and_record.py"
    spec = importlib.util.spec_from_file_location("step1_stream_and_record", step1_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load 1_stream_and_record.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_stacking_smoke() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot-smoke] SKIP: matplotlib not available ({exc})")
        return 0

    import numpy as np

    step1 = _load_step1_module()
    apply_lines = getattr(step1, "_apply_plot_lines", None)
    if apply_lines is None:
        raise RuntimeError("Missing _apply_plot_lines helper")

    fig, ax = plt.subplots()
    lines = []
    for _ in range(4):
        line, = ax.plot([], [], lw=1)
        lines.append(line)

    t = np.linspace(0.0, 1.0, 200)
    y = np.stack(
        [
            np.sin(2 * np.pi * t),
            np.sin(2 * np.pi * t + 0.5),
            np.sin(2 * np.pi * t + 1.0),
            np.sin(2 * np.pi * t + 1.5),
        ],
        axis=1,
    )
    offsets = np.arange(4, dtype=float) * 120.0
    apply_lines(lines, t, y, offsets, 4)

    assert len(lines) == 4, "Expected 4 Line2D objects"
    ys = [np.asarray(line.get_ydata(), dtype=float) for line in lines]
    assert all(arr.shape == t.shape for arr in ys), "Line ydata length mismatch"
    diff01 = float(np.nanmean(ys[1] - ys[0]))
    diff12 = float(np.nanmean(ys[2] - ys[1]))
    assert diff01 > 80.0 and diff12 > 80.0, "Channel offsets not applied"
    plt.close(fig)
    print("[plot-smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(plot_stacking_smoke())
