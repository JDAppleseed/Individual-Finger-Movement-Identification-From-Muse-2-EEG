#!/usr/bin/env python3
import json
from pathlib import Path

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("matplotlib required for figure generation") from e

    repo_root = Path(__file__).resolve().parent

    # Where LaTeX expects PDFs to live
    figs_dir = repo_root / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    out_dir = repo_root / "exercises" / "out"

    wrote = []

    # --- Fig: timebase effect (ex04) -> figs/timebase_alignment_effect.pdf ---
    p = out_dir / "ex04_timebase_break_and_measure.json"
    if p.exists():
        d = load_json(p)
        labels = ["aligned_kept", "offset_kept", "aligned_only", "offset_only"]
        vals = [float(d.get(k, 0) or 0) for k in labels]

        plt.figure()
        plt.bar(labels, vals)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("count")
        plt.title(f"Timebase misalignment impact (offset={d.get('offset_s')})")
        plt.tight_layout()

        out_path = figs_dir / "timebase_alignment_effect.pdf"
        plt.savefig(out_path)
        plt.close()
        wrote.append(out_path)

    # --- Fig: stability gate (ex09) -> figs/stability_gate_timeline.pdf ---
    p = out_dir / "ex09_stability_gate_demo.json"
    if p.exists():
        d = load_json(p)
        events = d.get("events", []) or []
        allow = [1 if (e.get("allow") is True) else 0 for e in events]

        plt.figure()
        plt.plot(allow)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("frame")
        plt.ylabel("allow (0/1)")
        plt.title("Stability gate allow/deny over time")
        plt.tight_layout()

        out_path = figs_dir / "stability_gate_timeline.pdf"
        plt.savefig(out_path)
        plt.close()
        wrote.append(out_path)

    if wrote:
        print("Wrote:")
        for p in wrote:
            print(" ", p)
    else:
        print("No figures written (missing exercises/out JSON outputs).")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())