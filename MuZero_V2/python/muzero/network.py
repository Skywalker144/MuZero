import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay import Outcome, PolicyHead, policy_head_count


def normalize_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
    minimum = hidden_state.amin(dim=[1, 2, 3], keepdim=True)
    maximum = hidden_state.amax(dim=[1, 2, 3], keepdim=True)
    span = maximum - minimum
    denominator = torch.where(span > 0, span, torch.ones_like(span))
    return (hidden_state - minimum) / denominator


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return tensor * scale + tensor.detach() * (1.0 - scale)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.silu(self.norm1(x)))
        out = self.conv2(F.silu(self.norm2(out)))
        return out + x


class RepresentationNet(nn.Module):
    def __init__(self, num_planes, num_channels, num_blocks):
        super().__init__()
        self.start_layer = nn.Sequential(
            nn.Conv2d(num_planes, num_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, num_channels), nn.SiLU(inplace=True),
        )
        self.trunk = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.start_layer(x)
        for block in self.trunk:
            x = block(x)
        return normalize_hidden_state(x)


class DynamicsNet(nn.Module):
    def __init__(self, board_size, num_channels, num_blocks):
        super().__init__()
        self.board_size = board_size
        self.start_layer = nn.Sequential(
            nn.Conv2d(num_channels + 1, num_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, num_channels), nn.SiLU(inplace=True),
        )
        self.trunk = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, hidden_state, action):
        action_plane = F.one_hot(action, self.board_size * self.board_size).to(hidden_state.dtype)
        action_plane = action_plane.reshape(-1, 1, self.board_size, self.board_size)
        x = self.start_layer(torch.cat([hidden_state, action_plane], dim=1))
        for block in self.trunk:
            x = block(x)
        return normalize_hidden_state(x)


class PredictionNet(nn.Module):
    def __init__(self, num_channels, value_head, auxiliary_policy_heads):
        super().__init__()
        self.wdl = value_head == 'wdl'
        self.policy_count = policy_head_count(auxiliary_policy_heads)
        self.win = int(Outcome.WIN)
        self.loss = int(Outcome.LOSS)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, num_channels), nn.SiLU(inplace=True),
            nn.Conv2d(num_channels, self.policy_count, kernel_size=1, bias=True),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, num_channels), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(num_channels, num_channels // 2), nn.SiLU(inplace=True),
            nn.Linear(num_channels // 2, len(Outcome) if self.wdl else 1),
        )

    def forward(self, hidden_state):
        return self.policy_head(hidden_state).flatten(2), self.value_head(hidden_state)

    def utility(self, logits: torch.Tensor) -> torch.Tensor:
        if self.wdl:
            probabilities = logits.softmax(-1)
            return probabilities[:, self.win] - probabilities[:, self.loss]
        return logits.squeeze(1).tanh()


class MuZeroNet(nn.Module):
    def __init__(self, board_size, num_planes, num_blocks, num_channels, value_head, auxiliary_policy_heads):
        super().__init__()
        self.board_size = board_size
        self.representation = RepresentationNet(num_planes, num_channels, num_blocks)
        self.dynamics = DynamicsNet(board_size, num_channels, num_blocks)
        self.prediction = PredictionNet(num_channels, value_head, auxiliary_policy_heads)


class InferenceModule(nn.Module):
    def __init__(self, network: MuZeroNet):
        super().__init__()
        self.network = network
        self.board_size = network.board_size
        self.protocol_version = 2
        self.main = int(PolicyHead.MAIN)

    @torch.jit.export
    def metadata(self) -> tuple[int, int]:
        return self.board_size, self.protocol_version

    @torch.jit.export
    def initial(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.network.representation(observation)
        policy, logits = self.network.prediction(hidden)
        return hidden, policy[:, self.main], self.network.prediction.utility(logits)

    @torch.jit.export
    def recurrent(self, hidden: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.network.dynamics(hidden, actions)
        policy, logits = self.network.prediction(hidden)
        return hidden, policy[:, self.main], self.network.prediction.utility(logits)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.initial(observation)
