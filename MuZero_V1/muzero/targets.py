from enum import IntEnum
from typing import TypedDict

import numpy as np


class PolicyHead(IntEnum):
    MAIN = 0
    SOFT = 1
    OPPONENT = 2
    SOFT_OPPONENT = 3


class Outcome(IntEnum):
    WIN = 0
    DRAW = 1
    LOSS = 2


class GameStep(TypedDict):
    observation: np.ndarray
    player: int
    action: int
    legal_actions: np.ndarray
    mcts_policy: np.ndarray
    policy_weight: float
    search_visits: int
    value_target: np.ndarray


class UnrollSample(TypedDict):
    observation: np.ndarray
    actions: np.ndarray
    policy_targets: np.ndarray
    value_targets: np.ndarray
    policy_mask: np.ndarray


def value_target(winner: int, player: int) -> np.ndarray:
    result = winner * player
    outcome = Outcome.WIN if result > 0 else Outcome.LOSS if result < 0 else Outcome.DRAW
    target = np.zeros(len(Outcome), dtype=np.float32)
    target[outcome] = 1
    return target


def soft_policy_target(policy: np.ndarray, legal_actions: np.ndarray) -> np.ndarray:
    target = np.zeros_like(policy, dtype=np.float32)
    target[legal_actions] = (policy[legal_actions] + 1e-7) ** 0.25
    return target / target.sum()
