"""
STEP 3b — Deepchecks Evaluation (SDS-aligned)
Deterministic model behavior (Dropout OFF)
"""

from pathlib import Path

import numpy as np
import torch
import joblib

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation, model_evaluation

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.label_schema import ACTION_NAMES
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    apply_channel_normalizer,
    summarize_windows,
)

# =========================
# ===== CONFIG ============
# =========================

SEED = 42

# Deterministic setup
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# ===== LOAD DATA =========
# =========================

X, y_action, y_finger, meta = load_sequence_npz("eeg_windows.npz")

# =========================
# ===== SAME SPLIT =========
# =========================

train_idx, test_idx = split_indices(y_action, y_finger, meta=meta, test_size=0.2, random_state=SEED)
X_train = X[train_idx]
X_test = X[test_idx]
y_train = y_action[train_idx]
y_test = y_action[test_idx]

# =========================
# ===== REUSE SCALER ======
# =========================

normalizer = joblib.load("scaler.save")
X_train = apply_channel_normalizer(X_train, normalizer)
X_test = apply_channel_normalizer(X_test, normalizer)

X_all = apply_channel_normalizer(X, normalizer)
# =========================
# ===== TABULAR SUMMARY ===
# =========================
# Deepchecks needs tabular data; we summarize windows here but
# run the model on raw window tensors via window_idx in predict().

train_df = summarize_windows(X_train)
train_df["window_idx"] = train_idx
test_df = summarize_windows(X_test)
test_df["window_idx"] = test_idx

assert "window_idx" in train_df.columns and "window_idx" in test_df.columns
assert test_idx.max() < len(X_all) and train_idx.max() < len(X_all)
assert y_action.min() >= 0
assert set(np.unique(y_action)).issubset(set(ACTION_NAMES.keys()))
assert max(ACTION_NAMES.keys()) >= int(y_action.max())

feature_names = [c for c in train_df.columns if c != "window_idx"]
class_names = [ACTION_NAMES[i] for i in sorted(ACTION_NAMES.keys())]

train_ds = Dataset(
    train_df,
    label=y_train,
    features=feature_names,
    label_type="multiclass",
    label_classes=class_names,
    cat_features=[],
)

test_ds = Dataset(
    test_df,
    label=y_test,
    features=feature_names,
    label_type="multiclass",
    label_classes=class_names,
    cat_features=[],
)

# =========================
# ===== MODEL (MATCH STEP 2)
# =========================

n_fingers = int(y_finger.max()) + 1
n_actions = int(y_action.max()) + 1

model = CNNLSTMFingerActionNet(n_channels=X.shape[2], n_fingers=n_fingers, n_actions=n_actions)
model_path = "finger_action_model.pt"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

class TorchModelWrapper:
    """
    Deepchecks-compatible wrapper.
    Returns deterministic action probabilities.
    """
    def predict(self, X_tabular):
        if "window_idx" not in X_tabular:
            raise KeyError("Deepchecks input is missing required column 'window_idx'.")
        window_idx = X_tabular["window_idx"].to_numpy().astype(np.int64)
        if (window_idx < 0).any() or (window_idx >= len(X_all)).any():
            raise ValueError("window_idx contains out-of-bounds indices for X_all.")
        device = next(model.parameters()).device
        X_tensor = torch.tensor(X_all[window_idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, action_logits = model(X_tensor)
            probs = torch.softmax(action_logits, dim=1)
        return probs.detach().cpu().numpy()

print("🔍 Running Deepchecks suites...")

suite = data_integrity().add(train_test_validation()).add(model_evaluation())

result = suite.run(train_ds, test_ds, model=TorchModelWrapper())

out_dir = Path("reports/deepchecks")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "deepchecks_eeg_report.html"
result.save_as_html(out_path.as_posix(), as_widget=False)
print(f"✅ Deepchecks report saved: {out_path}")
