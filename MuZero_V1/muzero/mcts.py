from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import SearchBudget, SearchConfig
from .targets import Outcome, PolicyHead
from .utils import add_dirichlet_noise, softmax


@dataclass(slots=True, eq=False)
class Node:
    to_play: int
    prior: float = 0.0
    parent: Node | None = None
    action_taken: int | None = None
    state: np.ndarray | None = None
    hidden_state: torch.Tensor | None = None
    children: list[Node] = field(default_factory=list)
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    visits: int = 0
    nn_value: float = 0.0

    def update(self, value: float):
        self.value_sum += value
        self.value_sq_sum += value * value
        self.visits += 1

    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    visit_policy: np.ndarray
    policy_target: np.ndarray
    root_value: float

    def play_policy(self, mode: str) -> np.ndarray:
        return self.policy_target if mode == "eval" else self.visit_policy


def select_child(node: Node, config: SearchConfig, cheap: bool = False) -> Node:
    mass = min(1.0, sum(child.prior for child in node.children if child.visits))
    mix = mass ** config.fpu_parent_weight_by_visited_policy_pow
    reduction = (
        config.root_fpu_reduction_max
        if node.parent is None and not cheap else config.fpu_reduction_max
    )
    fpu = mix * node.q_value() + (1 - mix) * node.nn_value - reduction * math.sqrt(mass)
    pb_c = config.pb_c_init + math.log((node.visits + config.pb_c_base + 1) / config.pb_c_base)
    pb_c *= math.sqrt(node.visits)
    best_score = -math.inf
    best_child = None
    for child in node.children:
        value = -child.q_value() if child.visits else fpu
        score = (value + 1) / 2 + pb_c * child.prior / (child.visits + 1)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


class Search:
    def __init__(self, game, state, to_play, config: SearchConfig, budget: SearchBudget, turn_number: int):
        if not isinstance(budget.visits, int) or budget.visits < 0:
            raise ValueError("search visits must be a nonnegative integer")
        if budget.visits == 0 and config.mode != "eval":
            raise ValueError("zero simulations are only supported in eval mode")
        self.game = game
        self.config = config
        self.budget = budget
        self.turn_number = turn_number
        self.root = Node(to_play, state=state)

    def next_leaf(self) -> Node | None:
        if self.root.visits == 0:
            return self.root
        if self.root.visits - 1 >= self.budget.visits:
            return None
        node = self.root
        while node.children:
            node = select_child(node, self.config, self.budget.cheap)
        return node

    def complete(self, node: Node, policy_logits: np.ndarray, value: float):
        if node is self.root:
            legal = self.game.get_legal_action_mask(node.state, node.to_play)
            logits = np.where(legal, policy_logits, -np.inf)
            noisy = self.config.mode == "train" and not self.budget.cheap
            if noisy:
                logits = logits / self.config.root_temperature(self.turn_number, self.game.board_size)
            policy = softmax(logits)
            if noisy:
                policy = add_dirichlet_noise(
                    policy, self.config.dirichlet_total_concentration, legal,
                    self.game.board_size, self.config.dirichlet_noise_weight,
                    shaped=self.config.shaped_dirichlet_noise,
                )
            actions = np.flatnonzero(legal)
        else:
            policy = softmax(policy_logits)
            actions = range(self.game.board_size ** 2)
        node.nn_value = value
        node.children = [
            Node(-node.to_play, prior=float(policy[action]), parent=node, action_taken=int(action))
            for action in actions
        ]
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    def result(self) -> SearchResult:
        weights = np.zeros(self.game.board_size ** 2, dtype=np.float64)
        for child in self.root.children:
            weights[child.action_taken] = child.visits if self.budget.visits else child.prior
        visit_policy = weights / weights.sum()
        if self.config.use_lcb_for_selection and self.budget.visits and not self.budget.cheap:
            children = self.root.children
            radius = np.full(len(children), 2.0 * self.config.lcb_stdevs)
            lcb = -radius.copy()
            for i, child in enumerate(children):
                if child.visits == 0:
                    continue
                n = float(child.visits)
                mean = child.q_value()
                prior_weight = 1.0 / (n * n)
                variance = max(0.0, child.value_sq_sum / n - mean * mean)
                variance += prior_weight / (n + prior_weight)
                effective_samples = (n + prior_weight) ** 2 / (n + prior_weight ** 2)
                radius[i] = self.config.lcb_stdevs * math.sqrt(variance / effective_samples)
                lcb[i] = -mean - radius[i]
            stable = max(children, key=lambda child: max(0, child.visits - 1) + 2 * child.prior)
            candidates = [
                i for i, child in enumerate(children)
                if child.visits > 0 and child.visits >= self.config.min_visit_prop_for_lcb * stable.visits
            ]
            best = max(candidates, key=lambda i: lcb[i])
            action = children[best].action_taken
            for i, child in enumerate(children):
                excess = lcb[best] - lcb[i]
                if i == best or child.visits == 0 or excess <= 0:
                    continue
                factor = (radius[i] + excess) / (radius[i] + 0.20 * excess)
                weights[action] = max(weights[action], factor * factor * child.visits)
        return SearchResult(visit_policy, weights / weights.sum(), self.root.q_value())

    def close(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            node.parent = None
            node.children = []
            node.hidden_state = None


class MCTS:
    def __init__(self, game, args, model, device):
        self.game = game
        self.config = SearchConfig.from_args(args, game.board_size)
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def infer_nodes(self, nodes: list[Node]) -> list[tuple[np.ndarray, float]]:
        hidden = [None] * len(nodes)
        root_indices = [i for i, node in enumerate(nodes) if node.parent is None]
        tree_indices = [i for i, node in enumerate(nodes) if node.parent is not None]
        if root_indices:
            observations = np.stack([
                self.game.encode_state(nodes[i].state, nodes[i].to_play) for i in root_indices
            ])
            states = self.model.representation(torch.as_tensor(
                observations, dtype=torch.float32, device=self.device
            ))
            for k, i in enumerate(root_indices):
                hidden[i] = states[k:k + 1]
        if tree_indices:
            parent_states = torch.cat([nodes[i].parent.hidden_state for i in tree_indices])
            actions = torch.tensor(
                [nodes[i].action_taken for i in tree_indices], dtype=torch.long, device=self.device
            )
            states = self.model.dynamics(parent_states, actions)
            for k, i in enumerate(tree_indices):
                hidden[i] = states[k:k + 1]
        policy_logits, value_logits = self.model.prediction(torch.cat(hidden))
        logits = policy_logits[:, PolicyHead.MAIN].float().cpu().numpy()
        wdl = value_logits.float().softmax(dim=-1)
        values = (wdl[:, Outcome.WIN] - wdl[:, Outcome.LOSS]).cpu().numpy()
        for node, state in zip(nodes, hidden):
            node.hidden_state = state
        return [(logits[i], float(values[i])) for i in range(len(nodes))]

    def search(self, state, to_play, num_simulations: int, turn_number: int = 0, cheap: bool = False) -> SearchResult:
        search = Search(
            self.game, state, to_play, self.config, SearchBudget(num_simulations, cheap), turn_number
        )
        try:
            while (node := search.next_leaf()) is not None:
                (policy, value), = self.infer_nodes([node])
                search.complete(node, policy, value)
            return search.result()
        finally:
            search.close()
