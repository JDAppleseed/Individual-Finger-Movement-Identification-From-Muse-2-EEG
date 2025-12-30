"""
STEP 2 — Train Multi-Head EEG Model (CNN + LSTM)
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    fit_channel_normalizer,
    apply_channel_normalizer,
)
from utils.experiment_logger import log_experiment, get_latest_experiment_hash
from utils.label_schema import ACTION_REST

SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
LOSS_ACTION_WEIGHT = 1.0

subject_id = "ANON"

try:
    EXP_HASH = get_latest_experiment_hash()
except Exception:
    meta_path = Path("session_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        subject_id = meta.get("subject_id", subject_id)
        EXP_HASH = meta.get("experiment_hash", "UNKNOWN")
    else:
        EXP_HASH = "UNKNOWN"

log_experiment(subject_id, EXP_HASH, "STEP_2_TRAIN")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EEGWindowDataset(Dataset):
    def __init__(self, X, y_finger, y_action):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_finger = torch.tensor(y_finger, dtype=torch.long)
        self.y_action = torch.tensor(y_action, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_finger[idx], self.y_action[idx]


set_seed(SEED)

X, y_action, y_finger, meta = load_sequence_npz("eeg_windows.npz")

train_idx, test_idx = split_indices(y_action, y_finger, meta=meta, test_size=0.2, random_state=SEED)
X_train, X_test = X[train_idx], X[test_idx]
y_action_train, y_action_test = y_action[train_idx], y_action[test_idx]
y_finger_train, y_finger_test = y_finger[train_idx], y_finger[test_idx]

normalizer = fit_channel_normalizer(X_train)
X_train = apply_channel_normalizer(X_train, normalizer)
X_test = apply_channel_normalizer(X_test, normalizer)
joblib.dump(normalizer, "scaler.save")

train_loader = DataLoader(
    EEGWindowDataset(X_train, y_finger_train, y_action_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)

test_loader = DataLoader(
    EEGWindowDataset(X_test, y_finger_test, y_action_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)

n_fingers = int(np.max(y_finger)) + 1
n_actions = int(np.max(y_action)) + 1

model = CNNLSTMFingerActionNet(n_channels=X.shape[2], n_fingers=n_fingers, n_actions=n_actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_f = nn.CrossEntropyLoss()
loss_a = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_action = 0
    total_finger = 0
    correct_action = 0
    correct_finger = 0

    for Xb, yfb, yab in train_loader:
        Xb = Xb.to(device)
        yfb = yfb.to(device)
        yab = yab.to(device)

        opt.zero_grad()
        f_out, a_out = model(Xb)

        loss_action = loss_a(a_out, yab)
        mask = yab != ACTION_REST
        if mask.any():
            loss_finger = loss_f(f_out[mask], yfb[mask])
        else:
            loss_finger = torch.tensor(0.0, device=device)

        loss = loss_action + LOSS_ACTION_WEIGHT * loss_finger
        loss.backward()
        opt.step()

        total_loss += loss.item() * Xb.size(0)
        preds_action = torch.argmax(a_out, dim=1)
        correct_action += (preds_action == yab).sum().item()
        total_action += yab.numel()

        if mask.any():
            preds_finger = torch.argmax(f_out[mask], dim=1)
            correct_finger += (preds_finger == yfb[mask]).sum().item()
            total_finger += yfb[mask].numel()

    avg_loss = total_loss / max(1, len(train_loader.dataset))
    action_acc = correct_action / max(1, total_action)
    finger_acc = correct_finger / max(1, total_finger)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | loss={avg_loss:.4f} "
            f"action_acc={action_acc:.3f} finger_acc={finger_acc:.3f}"
        )

model.eval()

# Save model and config
model_path = "finger_action_model.pt"
torch.save(model.state_dict(), model_path)

train_config = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "loss_action_weight": LOSS_ACTION_WEIGHT,
    "n_fingers": n_fingers,
    "n_actions": n_actions,
    "input_shape": list(X.shape[1:]),
    "normalizer": {"type": normalizer["type"], "channels": normalizer["channels"]},
    "device": str(device),
    "model": "CNNLSTMFingerActionNet",
}
config_path = Path("logs") / "experiments" / f"{EXP_HASH}_train_config.json"
config_path.write_text(json.dumps(train_config, indent=2))

# Save test predictions for reproducibility
all_action_probs = []
all_finger_probs = []
with torch.no_grad():
    for Xb, yfb, yab in test_loader:
        Xb = Xb.to(device)
        f_out, a_out = model(Xb)
        all_finger_probs.append(torch.softmax(f_out, dim=1).cpu().numpy())
        all_action_probs.append(torch.softmax(a_out, dim=1).cpu().numpy())

np.savez_compressed(
    "test_predictions.npz",
    action_probs=np.concatenate(all_action_probs, axis=0),
    finger_probs=np.concatenate(all_finger_probs, axis=0),
    y_action=y_action_test,
    y_finger=y_finger_test,
    test_indices=test_idx,
)

log_experiment(subject_id, EXP_HASH, "STEP_2_COMPLETE", f"loss={avg_loss:.4f}")
print("✅ Training complete")
