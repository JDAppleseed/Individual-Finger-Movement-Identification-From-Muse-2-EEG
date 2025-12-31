#!/usr/bin/env python3
"""
Minimal CLI regression checks for 2_train_model.py.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd, env):
    proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def main():
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable
    script_path = repo_root / "2_train_model.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        code, out = run([python_exe, str(script_path), "--help"], cwd=tmp_path, env=env)
        print(out.strip())
        if code != 0:
            print("FAIL: --help returned nonzero")
            return 1
        if (tmp_path / "finger_action_model.pt").exists():
            print("FAIL: model artifact created during --help")
            return 1

        npz_path = repo_root / "eeg_windows.npz"
        if not npz_path.exists():
            print("SKIP: eeg_windows.npz missing")
            return 0

        model_out = tmp_path / "finger_action_model.pt"
        scaler_out = tmp_path / "scaler.save"
        preds_out = tmp_path / "test_predictions.npz"

        cmd = [
            python_exe,
            str(script_path),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--npz",
            str(npz_path),
            "--save-model",
            str(model_out),
            "--save-scaler",
            str(scaler_out),
            "--save-preds",
            str(preds_out),
        ]
        code, out = run(cmd, cwd=repo_root, env=env)
        print(out.strip())
        if code != 0:
            print("FAIL: training run failed")
            return 1

        print("✅ self_check_train_cli passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
