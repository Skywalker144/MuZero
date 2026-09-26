import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import Plane, VERSION
from .replay import Outcome, PolicyHead, policy_head_count


def normalize_hidden_state(hidden_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    minimum = hidden_state.masked_fill(mask == 0, float('inf')).amin(dim=[1, 2, 3], keepdim=True)
    maximum = hidden_state.masked_fill(mask == 0, float('-inf')).amax(dim=[1, 2, 3], keepdim=True)
    span = maximum - minimum
    denominator = torch.where(span > 0, span, torch.ones_like(span))
    return ((hidden_state - minimum) / denominator) * mask


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return tensor * scale + tensor.detach() * (1.0 - scale)


class MaskedNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=[2, 3], keepdim=True) * x.size(1)
        mean = (x * mask).sum(dim=[1, 2, 3], keepdim=True) / count
        variance = ((x - mean).square() * mask).sum(dim=[1, 2, 3], keepdim=True) / count
        return ((x - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias) * mask


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = MaskedNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = MaskedNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.silu(self.norm1(x, mask))) * mask
        out = self.conv2(F.silu(self.norm2(out, mask))) * mask
        return out + x


class RepresentationNet(nn.Module):
    def __init__(self, num_planes, num_channels, num_blocks):
        super().__init__()
        self.mask_plane = int(Plane.ON_BOARD)
        self.start_layer = nn.Conv2d(num_planes, num_channels, kernel_size=3, padding=1, bias=False)
        self.norm = MaskedNorm(num_channels)
        self.trunk = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, self.mask_plane:self.mask_plane + 1]
        x = F.silu(self.norm(self.start_layer(x * mask), mask))
        for block in self.trunk:
            x = block(x, mask)
        return torch.cat([normalize_hidden_state(x, mask), mask], dim=1)


class DynamicsNet(nn.Module):
    def __init__(self, canvas_size, num_channels, num_blocks):
        super().__init__()
        self.canvas_size = canvas_size
        self.start_layer = nn.Conv2d(num_channels + 1, num_channels, kernel_size=3, padding=1, bias=False)
        self.norm = MaskedNorm(num_channels)
        self.trunk = nn.ModuleList([ResBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mask = hidden_state[:, -1:]
        action_plane = F.one_hot(action, self.canvas_size * self.canvas_size).to(hidden_state.dtype)
        action_plane = action_plane.reshape(-1, 1, self.canvas_size, self.canvas_size) * mask
        x = self.start_layer(torch.cat([hidden_state[:, :-1] * mask, action_plane], dim=1))
        x = F.silu(self.norm(x, mask))
        for block in self.trunk:
            x = block(x, mask)
        return torch.cat([normalize_hidden_state(x, mask), mask], dim=1)


class PredictionNet(nn.Module):
    def __init__(self, num_channels, value_head, auxiliary_policy_heads):
        super().__init__()
        self.wdl = value_head == 'wdl'
        self.policy_count = policy_head_count(auxiliary_policy_heads)
        self.win = int(Outcome.WIN)
        self.loss = int(Outcome.LOSS)
        self.policy_projection = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.policy_norm = MaskedNorm(num_channels)
        self.policy_head = nn.Conv2d(num_channels, self.policy_count, kernel_size=1, bias=True)
        self.value_projection = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.value_norm = MaskedNorm(num_channels)
        self.value_head = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2), nn.SiLU(inplace=True),
            nn.Linear(num_channels // 2, len(Outcome) if self.wdl else 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = hidden_state[:, :-1], hidden_state[:, -1:]
        policy = self.policy_head(F.silu(self.policy_norm(self.policy_projection(x), mask)))
        policy = policy.masked_fill(mask == 0, -1e9).flatten(2)
        value = F.silu(self.value_norm(self.value_projection(x), mask))
        pooled = (value * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3])
        return policy, self.value_head(pooled)

    def utility(self, logits: torch.Tensor) -> torch.Tensor:
        if self.wdl:
            probabilities = logits.softmax(-1)
            return probabilities[:, self.win] - probabilities[:, self.loss]
        return logits.squeeze(1).tanh()


class MuZeroNet(nn.Module):
    def __init__(self, canvas_size, num_planes, num_blocks, num_channels, value_head, auxiliary_policy_heads):
        super().__init__()
        self.canvas_size = canvas_size
        self.representation = RepresentationNet(num_planes, num_channels, num_blocks)
        self.dynamics = DynamicsNet(canvas_size, num_channels, num_blocks)
        self.prediction = PredictionNet(num_channels, value_head, auxiliary_policy_heads)


class InferenceModule(nn.Module):
    def __init__(self, network: MuZeroNet):
        super().__init__()
        self.network = network
        self.canvas_size = network.canvas_size
        self.protocol_version = VERSION
        self.main = int(PolicyHead.MAIN)

    @torch.jit.export
    def metadata(self) -> tuple[int, int]:
        return self.canvas_size, self.protocol_version

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
