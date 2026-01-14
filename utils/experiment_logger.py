"""
Experiment Logger
-----------------
• Generates anonymous subject IDs
• Creates deterministic experiment hashes
• Logs metadata safely
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

try:
    import fcntl  # Unix/macOS
except ImportError:
    fcntl = None


ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "utils" / "subject_registry.json"
LOG_DIR = ROOT / "logs" / "experiments"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# ===== SUBJECT ID ========
# =========================


def get_subject_id(gender: str, age: int):
    gender = gender.upper()
    key = f"{gender}{age}"

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text("{}")

    with open(REGISTRY_PATH, "r+") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)

        registry = json.load(f)
        count = registry.get(key, 0) + 1
        registry[key] = count

        f.seek(0)
        json.dump(registry, f, indent=2)
        f.truncate()

        if fcntl:
            fcntl.flock(f, fcntl.LOCK_UN)

    return f"{count}-{gender}{age}"


# =========================
# ===== HASH ==============
# =========================


def generate_experiment_hash(subject_id: str, config_dict: dict):
    """
    Deterministic hash:
    Same subject + same config → same hash
    """
    payload = {"subject_id": subject_id, "config": config_dict}
    serialized = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:12]


# =========================
# ===== LOGGING ===========
# =========================


def log_experiment(subject_id, exp_hash, step, notes=None):
    log_path = LOG_DIR / f"{exp_hash}.json"

    entry = {
        "step": step,
        "datetime": datetime.utcnow().isoformat(),
        "notes": notes or "",
    }

    if log_path.exists():
        data = json.loads(log_path.read_text())
        data["steps"].append(entry)
    else:
        data = {"experiment_hash": exp_hash, "subject_id": subject_id, "steps": [entry]}

    log_path.write_text(json.dumps(data, indent=2))


# =========================
# ===== UTIL ==============
# =========================


def get_latest_experiment_hash():
    logs = sorted(LOG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise RuntimeError("No experiment logs found")
    return logs[-1].stem
