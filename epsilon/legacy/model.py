from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from action_space import ACTION_SIZE


class LinithNet(nn.Module):
    """
    Combined policy + value network for Linith.

    Input:
      x: (batch, 6, 10, 10)

    Outputs:
      policy_logits: (batch, ACTION_SIZE) – unnormalized log-probabilities over actions
      value:         (batch, 1)          – in [-1, 1], estimated outcome for current player
    """

    def __init__(self):
        super().__init__()

        # shared trunk
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 64 channels * 10 * 10 = 6400 features
        self.fc_shared = nn.Linear(64 * 10 * 10, 256)

        # policy head
        self.fc_policy = nn.Linear(256, ACTION_SIZE)

        # value head
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, 6, 10, 10)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)          # (B, 6400)
        h = F.relu(self.fc_shared(x))      # (B, 256)

        policy_logits = self.fc_policy(h)  # (B, ACTION_SIZE)
        value_raw = self.fc_value(h)       # (B, 1)
        value = torch.tanh(value_raw)      # clamp to [-1, 1]

        return policy_logits, value
