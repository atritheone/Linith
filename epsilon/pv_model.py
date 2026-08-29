import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .action_space import ACTION_SIZE
except ImportError:
    from action_space import ACTION_SIZE

NUM_ACTIONS = ACTION_SIZE


class LinithPVNet(nn.Module):
    """
    Policy + Value network for Linith.

    Input:  [B, C, 10, 10]
    Output:
      - policy_logits: [B, NUM_ACTIONS]
      - value:         [B, 1] in [-1, 1]
    """

    def __init__(self, board_channels: int = 8, board_size: int = 10):
        super().__init__()
        self.board_size = board_size
        self.board_channels = board_channels

        # trunk
        self.conv1 = nn.Conv2d(board_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # policy head
        self.policy_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(64)
        self.policy_fc = nn.Linear(64 * board_size * board_size, NUM_ACTIONS)

        # value head
        self.value_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(64)
        self.value_fc1 = nn.Linear(64 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        returns:
          policy_logits: [B, NUM_ACTIONS]
          value:         [B, 1]
        """
        h = self.trunk(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # value
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return policy_logits, v

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load current or legacy 6-channel/752-action checkpoints safely."""
        adapted = state_dict.copy()
        current = super().state_dict()
        for key in ("conv1.weight", "policy_fc.weight", "policy_fc.bias"):
            source = adapted.get(key)
            target = current.get(key)
            if source is None or target is None or source.shape == target.shape:
                continue
            merged = target.clone()
            if key == "conv1.weight" and source.ndim == target.ndim == 4:
                bounds = tuple(min(a, b) for a, b in zip(source.shape, target.shape))
                slices = tuple(slice(0, n) for n in bounds)
                merged[slices] = source[slices]
            elif key.startswith("policy_fc"):
                # Placement/group indices 0..703 retained their exact meaning.
                rows = min(704, source.shape[0], target.shape[0])
                if source.ndim == 2:
                    merged[:rows, :min(source.shape[1], target.shape[1])] = source[
                        :rows, :min(source.shape[1], target.shape[1])
                    ]
                else:
                    merged[:rows] = source[:rows]
            adapted[key] = merged
        try:
            return super().load_state_dict(adapted, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(adapted, strict=strict)

    @staticmethod
    def from_value_only(value_model_path: str, board_channels: int = 8, board_size: int = 10):
        """
        Try to load weights from an older value-only model.
        Missing keys (policy head) will remain randomly initialised.
        """
        net = LinithPVNet(board_channels=board_channels, board_size=board_size)
        try:
            state = torch.load(value_model_path, map_location="cpu")
            missing, unexpected = net.load_state_dict(state, strict=False)
            print("[pv_model] Loaded value-only weights from", value_model_path)
            print("[pv_model] Missing keys (policy head):", missing)
            print("[pv_model] Unexpected keys:", unexpected)
        except Exception as e:
            print("[pv_model] WARNING: could not load value-only weights:", e)
        return net
