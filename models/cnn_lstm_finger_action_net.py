import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMFingerActionNet(nn.Module):
    """
    CNN + LSTM multi-head classifier for EEG windows.
    Input: [B, T, C] where C=channels.
    Outputs: finger_logits, action_logits
    """

    def __init__(
        self,
        n_channels=4,
        n_fingers=6,
        n_actions=3,
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

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # [B, F, T] -> [B, T, F]
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        features = out[:, -1, :]
        features = self.head_dropout(features)
        finger_logits = self.finger_head(features)
        action_logits = self.action_head(features)
        return finger_logits, action_logits

    @torch.no_grad()
    def deterministic_forward(self, x):
        self.eval()
        finger_logits, action_logits = self.forward(x)
        finger_probs = F.softmax(finger_logits, dim=-1)
        action_probs = F.softmax(action_logits, dim=-1)
        return finger_probs, action_probs

    @torch.no_grad()
    def mc_forward(self, x, passes=20):
        if passes < 1:
            raise ValueError("passes must be >= 1")
        was_training = self.training
        self.train()
        finger_probs = []
        action_probs = []
        for _ in range(passes):
            f_logits, a_logits = self.forward(x)
            finger_probs.append(F.softmax(f_logits, dim=-1))
            action_probs.append(F.softmax(a_logits, dim=-1))

        finger_stack = torch.stack(finger_probs, dim=0)
        action_stack = torch.stack(action_probs, dim=0)

        finger_mean = finger_stack.mean(dim=0)
        action_mean = action_stack.mean(dim=0)
        finger_std = finger_stack.std(dim=0)
        action_std = action_stack.std(dim=0)

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

        return {
            "finger_mean": finger_mean,
            "action_mean": action_mean,
            "finger_std": finger_std,
            "action_std": action_std,
            "finger_entropy": finger_entropy,
            "action_entropy": action_entropy,
            "finger_mi": finger_mi,
            "action_mi": action_mi,
        }
