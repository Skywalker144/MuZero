import math
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SearchBudget:
    visits: int
    cheap: bool = False

    @property
    def policy_weight(self) -> float:
        return float(not self.cheap)


@dataclass(frozen=True)
class SearchConfig:
    num_simulations: int
    cheap_search_visits: int
    dirichlet_total_concentration: float
    mode: Literal["train", "eval"] = "train"
    cheap_search_prob: float = 0.75
    pb_c_init: float = 1.25
    pb_c_base: float = 19652
    dirichlet_noise_weight: float = 0.25
    shaped_dirichlet_noise: bool = True
    use_lcb_for_selection: bool = True
    lcb_stdevs: float = 5.0
    min_visit_prop_for_lcb: float = 0.15
    chosen_move_temperature_early: float = 0.75
    chosen_move_temperature: float = 0.15
    chosen_move_temperature_halflife: float = 19.0
    root_policy_temperature_early: float = 1.3
    root_policy_temperature: float = 1.1
    fpu_reduction_max: float = 0.2
    root_fpu_reduction_max: float = 0.0
    fpu_parent_weight_by_visited_policy_pow: float = 2.0

    @classmethod
    def from_args(cls, args: dict, board_size: int) -> "SearchConfig":
        defaults = {
            "num_simulations": max(1, int(1.7 * board_size ** 2)),
            "cheap_search_visits": max(1, int(0.28 * board_size ** 2)),
            "dirichlet_total_concentration": 0.03 * board_size ** 2,
        }
        names = {item.name for item in fields(cls)}
        return cls(**(defaults | {key: value for key, value in args.items() if key in names}))

    def __post_init__(self):
        if self.mode not in ("train", "eval"):
            raise ValueError("mode must be train or eval")
        for name in ("num_simulations", "cheap_search_visits"):
            value = getattr(self, name)
            minimum = 0 if name == "num_simulations" and self.mode == "eval" else 1
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("cheap_search_prob", "dirichlet_noise_weight", "min_visit_prop_for_lcb"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "pb_c_base", "dirichlet_total_concentration", "chosen_move_temperature_halflife",
            "root_policy_temperature_early", "root_policy_temperature",
            "fpu_parent_weight_by_visited_policy_pow",
            "lcb_stdevs",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "pb_c_init", "chosen_move_temperature_early", "chosen_move_temperature",
            "fpu_reduction_max", "root_fpu_reduction_max",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def sample_budget(self) -> SearchBudget:
        cheap = self.mode == "train" and np.random.random() < self.cheap_search_prob
        visits = min(self.num_simulations, self.cheap_search_visits) if cheap else self.num_simulations
        return SearchBudget(visits, cheap)

    def interpolate(self, early: float, late: float, turn: int, board_size: int) -> float:
        halflives = turn * 19.0 / (self.chosen_move_temperature_halflife * board_size)
        return late + (early - late) * 0.5 ** halflives

    def move_temperature(self, turn: int, board_size: int) -> float:
        if self.mode == "eval":
            return 0.0
        return self.interpolate(
            self.chosen_move_temperature_early, self.chosen_move_temperature, turn, board_size
        )

    def root_temperature(self, turn: int, board_size: int) -> float:
        return self.interpolate(
            self.root_policy_temperature_early, self.root_policy_temperature, turn, board_size
        )


@dataclass(frozen=True)
class LossConfig:
    value_loss_scale: float = 1.2
    soft_policy_loss_scale: float = 0.5
    opponent_policy_loss_scale: float = 0.15

    @classmethod
    def from_args(cls, args: dict) -> "LossConfig":
        return cls(**{item.name: args[item.name] for item in fields(cls) if item.name in args})

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be finite and nonnegative")
