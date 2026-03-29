import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMFingerActionNet(nn.Module):
    """
    CNN + LSTM multi-head classifier for EEG windows.
    Input: [B, T, C] where C=channels.
    Outputs: finger_logits, action_logits, applicability_logit (optional)
    """

    def __init__(
        self,
        n_channels=4,
        n_fingers=6,
        n_actions=3,
        finger_applicability_head=False,
        conv_channels=(16, 32),
        kernel_sizes=(5, 3),
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.3,
        group_norm_groups=4,
    ):
        super().__init__()
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must match length")

        conv_layers = []
        in_ch = n_channels
        for out_ch, k in zip(conv_channels, kernel_sizes):
            conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            conv_layers.append(
                nn.GroupNorm(
                    num_groups=min(group_norm_groups, out_ch), num_channels=out_ch
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(p=dropout))
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(p=dropout)
        self.finger_head = nn.Linear(lstm_hidden, n_fingers)
        self.action_head = nn.Linear(lstm_hidden, n_actions)
        self.finger_applicability_head = (
            nn.Linear(lstm_hidden, 1) if bool(finger_applicability_head) else None
        )

    def _encode_sequence(self, x, state=None):
        # x: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # [B, F, T] -> [B, T, F]
        x = x.permute(0, 2, 1)
        out, next_state = self.lstm(x, state)
        features = out[:, -1, :]
        features = self.head_dropout(features)
        return features, next_state

    def extract_features(self, x):
        features, _ = self._encode_sequence(x)
        return features

    def forward_heads(self, features):
        finger_logits = self.finger_head(features)
        action_logits = self.action_head(features)
        applicability_logits = None
        if self.finger_applicability_head is not None:
            applicability_logits = self.finger_applicability_head(features).squeeze(-1)
        return finger_logits, action_logits, applicability_logits

    def forward_with_state(self, x, state=None):
        features, next_state = self._encode_sequence(x, state)
        finger_logits, action_logits, applicability_logits = self.forward_heads(features)
        return finger_logits, action_logits, applicability_logits, next_state

    def forward(self, x):
        features = self.extract_features(x)
        return self.forward_heads(features)

    @torch.no_grad()
    def deterministic_forward(self, x):
        self.eval()
        finger_logits, action_logits, applicability_logits = self.forward(x)
        finger_probs = F.softmax(finger_logits, dim=-1)
        action_probs = F.softmax(action_logits, dim=-1)
        applicability_probs = None
        if applicability_logits is not None:
            applicability_probs = torch.sigmoid(applicability_logits)
        return finger_probs, action_probs, applicability_probs

    @torch.no_grad()
    def mc_forward(self, x, passes=20):
        if passes < 1:
            raise ValueError("passes must be >= 1")
        was_training = self.training
        self.train()
        batch_size = int(x.shape[0])
        mc_input = x.repeat((passes, 1, 1)) if passes > 1 else x
        f_logits, a_logits, app_logits = self.forward(mc_input)
        finger_stack = F.softmax(f_logits, dim=-1).reshape(passes, batch_size, -1)
        action_stack = F.softmax(a_logits, dim=-1).reshape(passes, batch_size, -1)

        finger_mean = finger_stack.mean(dim=0)
        action_mean = action_stack.mean(dim=0)
        if passes > 1:
            finger_std = finger_stack.std(dim=0)
            action_std = action_stack.std(dim=0)
        else:
            finger_std = torch.zeros_like(finger_mean)
            action_std = torch.zeros_like(action_mean)

        finger_entropy = -torch.sum(finger_mean * torch.log(finger_mean + 1e-8), dim=-1)
        action_entropy = -torch.sum(action_mean * torch.log(action_mean + 1e-8), dim=-1)
        finger_exp_entropy = -torch.sum(
            finger_stack * torch.log(finger_stack + 1e-8), dim=-1
        ).mean(dim=0)
        action_exp_entropy = -torch.sum(
            action_stack * torch.log(action_stack + 1e-8), dim=-1
        ).mean(dim=0)

        finger_mi = finger_entropy - finger_exp_entropy
        action_mi = action_entropy - action_exp_entropy

        if not was_training:
            self.eval()

        result = {
            "finger_mean": finger_mean,
            "action_mean": action_mean,
            "finger_std": finger_std,
            "action_std": action_std,
            "finger_entropy": finger_entropy,
            "action_entropy": action_entropy,
            "finger_mi": finger_mi,
            "action_mi": action_mi,
        }
        if app_logits is not None:
            applicability_stack = torch.sigmoid(app_logits)
            if applicability_stack.ndim == 1:
                applicability_stack = applicability_stack.reshape(passes, batch_size)
            else:
                applicability_stack = applicability_stack.reshape(
                    passes, batch_size, -1
                )
            result["applicability_mean"] = applicability_stack.mean(dim=0)
            if passes > 1:
                result["applicability_std"] = applicability_stack.std(dim=0)
            else:
                result["applicability_std"] = torch.zeros_like(
                    result["applicability_mean"]
                )
        return result
