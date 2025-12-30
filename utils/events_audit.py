"""
Lightweight edit log for event corrections.
"""

import json
from pathlib import Path
from datetime import datetime


def log_event_edit(action: str, before: dict, after: dict, note: str = ""):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "events_edits.txt"

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "note": note,
        "before": before,
        "after": after,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
