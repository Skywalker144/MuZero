import numpy as np

from .config import SearchBudget, SearchConfig
from .mcts import SearchResult
from .targets import GameStep, Outcome, value_target
from .utils import apply_temperature


class GameHistory:
    def __init__(self, game, config: SearchConfig):
        self.game = game
        self.config = config
        self.state = game.get_initial_state()
        self.to_play = 1
        self.steps: list[GameStep] = []
        self.winner: int | None = None

    def play(self, result: SearchResult, budget: SearchBudget):
        temperature = self.config.move_temperature(len(self.steps), self.game.board_size)
        move_policy = apply_temperature(result.play_policy(self.config.mode), temperature)
        action = int(np.argmax(move_policy)) if temperature == 0 else int(
            np.random.choice(len(move_policy), p=move_policy)
        )
        self.steps.append({
            "observation": self.game.encode_state(self.state, self.to_play),
            "player": self.to_play,
            "action": action,
            "legal_actions": self.game.get_legal_action_mask(self.state, self.to_play),
            "mcts_policy": result.policy_target,
            "policy_weight": budget.policy_weight,
            "search_visits": budget.visits,
            "value_target": np.zeros(len(Outcome), dtype=np.float32),
        })
        self.state = self.game.get_next_state(self.state, action, self.to_play)
        self.to_play = -self.to_play
        winner = self.game.get_winner(self.state, self.to_play)
        if winner is not None:
            self.winner = int(winner)
            for step in self.steps:
                step["value_target"] = value_target(self.winner, step["player"])

    def result(self) -> tuple[list[GameStep], int, int]:
        if self.winner is None:
            raise ValueError("the game has not finished")
        return self.steps, self.winner, len(self.steps)
