# models/finger_action_net.py

import torch.nn as nn


class FingerActionNet(nn.Module):
    """
    CNN-based multi-head EEG classifier
    Outputs:
      - Finger identity (Thumb–Pinky)
      - Action state (e.g. open / close / rest)

    Supports:
      - Deterministic inference
      - MC Dropout for Bayesian uncertainty
    """

    def __init__(
        self,
        n_channels: int = 4,
        window_samples: int = 64,
        n_fingers: int = 6,
        n_actions: int = 3,
        finger_applicability_head: bool = False,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        # =========================
        # Shared CNN Encoder
        # =========================
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.AdaptiveAvgPool1d(1),  # shape → (B, 32, 1)
        )

        self.latent_dim = 32

        # =========================
        # Multi-Head Outputs
        # =========================
        self.finger_head = nn.Linear(self.latent_dim, n_fingers)
        self.action_head = nn.Linear(self.latent_dim, n_actions)
        self.finger_applicability_head = (
            nn.Linear(self.latent_dim, 1) if bool(finger_applicability_head) else None
        )

    def _prepare_input(self, x):
        """
        x shape:
          - (B, C, T) preferred
          - (B, T, C) accepted when C=4
          - (B, C*T) fallback (auto-reshaped)
        """

        if x.ndim == 2:
            # Fallback compatibility with earlier scripts
            B = x.shape[0]
            x = x.view(B, 4, -1)
        elif x.ndim == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.permute(0, 2, 1)
        return x

    def extract_features(self, x):
        x = self._prepare_input(x)
        return self.encoder(x).squeeze(-1)

    def forward_heads(self, features):
        finger_logits = self.finger_head(features)
        action_logits = self.action_head(features)
        applicability_logits = None
        if self.finger_applicability_head is not None:
            applicability_logits = self.finger_applicability_head(features).squeeze(-1)
        return finger_logits, action_logits, applicability_logits

    def forward(self, x):
        z = self.extract_features(x)
        return self.forward_heads(z)
