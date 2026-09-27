from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    hidden_state_num_channels: int
    representation_num_blocks: int
    representation_num_channels: int
    dynamics_num_blocks: int
    dynamics_num_channels: int
    prediction_backbone_num_blocks: int
    prediction_backbone_num_channels: int
    policy_head_num_blocks: int
    policy_head_num_channels: int
    value_head_num_blocks: int
    value_head_num_channels: int
    value_head_hidden_channels: int
    value_head: str
    auxiliary_policy_heads: bool

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> 'ModelConfig':
        return cls(**{field.name: config[field.name.upper()] for field in fields(cls)})


MODEL_KEYS = tuple(field.name.upper() for field in fields(ModelConfig))
MODEL_INT_KEYS = {field.name.upper() for field in fields(ModelConfig) if field.type is int}
