#!/usr/bin/env python3
import json
from pathlib import Path

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("matplotlib required for figure generation") from e

    root = Path(__file__).resolve().parent
    out = root
    out.mkdir(parents=True, exist_ok=True)

    # --- Fig: timebase effect (ex04) ---
    p = Path("exercises/out/ex04_timebase_break_and_measure.json")
    if p.exists():
        d = load(p)
        labels = ["aligned_kept","offset_kept","aligned_only","offset_only"]
        vals = [d.get(k,0) for k in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("count")
        plt.title(f"Timebase misalignment impact (offset={d.get('offset_s')})")
        plt.tight_layout()
        plt.savefig(out / "timebase_alignment_effect.pdf")
        plt.close()

    # --- Fig: stability gate (ex09) ---
    p = Path("exercises/out/ex09_stability_gate_demo.json")
    if p.exists():
        d = load(p)
        events = d.get("events", [])
        allow = [1 if e.get("allow") else 0 for e in events]
        plt.figure()
        plt.plot(allow)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("frame")
        plt.ylabel("allow (0/1)")
        plt.title("Stability gate allow/deny over time")
        plt.tight_layout()
        plt.savefig(out / "stability_gate_timeline.pdf")
        plt.close()

    print("wrote figures into figs/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
