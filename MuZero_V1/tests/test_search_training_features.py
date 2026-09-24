import numpy as np
import pytest
import torch

from envs.tictactoe import TicTacToe
from muzero.config import SearchBudget, SearchConfig
from muzero.mcts import MCTS, Node, Search, SearchResult, select_child
from muzero.network import MuZeroNet
from muzero.replay_buffer import ReplayBuffer
from muzero.selfplay import GameHistory
from muzero.targets import PolicyHead, value_target
from muzero.utils import apply_temperature, random_augment_batch


def trajectory():
    game = TicTacToe()
    state = game.get_initial_state()
    records = []
    for t, action in enumerate([0, 3, 1, 4, 2]):
        player = (-1) ** t
        legal = game.get_legal_action_mask(state, player)
        policy = np.zeros(9, dtype=np.float32)
        policy[action] = 1
        records.append({
            "observation": game.encode_state(state, player),
            "player": player,
            "action": action,
            "legal_actions": legal,
            "mcts_policy": policy,
            "policy_weight": float(t != 1),
            "search_visits": 2 if t == 1 else 8,
            "value_target": value_target(1, player),
        })
        state = game.get_next_state(state, action, player)
    return records


def test_temperature_preserves_support_and_limits():
    policy = np.array([0.8, 0.2, 0.0])
    np.testing.assert_allclose(apply_temperature(policy, 2), [2 / 3, 1 / 3, 0])
    np.testing.assert_array_equal(apply_temperature(policy, 0), [1, 0, 0])
    np.testing.assert_array_equal(apply_temperature(policy, 1), policy)
    assert np.isfinite(apply_temperature(policy, 1e-5)).all()


def test_temperature_schedule_has_board_scaled_half_life():
    config = SearchConfig.from_args({}, 9)
    assert config.move_temperature(0, 9) == pytest.approx(0.75)
    assert config.move_temperature(9, 9) == pytest.approx(0.45)
    assert config.root_temperature(9, 9) == pytest.approx(1.2)
    assert config.move_temperature(10000, 9) == pytest.approx(0.15)


@pytest.mark.parametrize("cheap_prob, visits, weight", [(0, 8, 1), (1, 2, 0)])
def test_budget_randomization_extremes(cheap_prob, visits, weight):
    config = SearchConfig.from_args({
        "num_simulations": 8, "cheap_search_visits": 2,
        "cheap_search_prob": cheap_prob,
    }, 3)
    budget = config.sample_budget()
    assert budget.visits == visits
    assert budget.policy_weight == weight
    evaluation = SearchConfig.from_args({
        "mode": "eval", "num_simulations": 8, "cheap_search_prob": 1,
    }, 3)
    assert evaluation.sample_budget() == SearchBudget(8, False)


def test_fpu_mixes_parent_and_network_in_parent_scale():
    config = SearchConfig.from_args({"pb_c_init": 0, "pb_c_base": 1e100}, 3)
    root = Node(1)
    root.nn_value = 0.8
    root.value_sum = -0.8
    root.visits = 2
    visited = Node(-1, prior=0.5, parent=root, action_taken=0)
    visited.update(-0.4)
    unseen = Node(-1, prior=0.5, parent=root, action_taken=1)
    root.children = [visited, unseen]
    assert select_child(root, config, cheap=False) is unseen
    assert select_child(root, config, cheap=True) is visited
    root.parent = Node(-1)
    assert select_child(root, config, cheap=False) is visited


@pytest.mark.parametrize("mode, cheap, expected", [
    ("train", False, [0, 0.5, 0.5]),
    ("train", True, [0, 0.9, 0.1]),
    ("eval", False, [0, 0.9, 0.1]),
])
def test_root_temperature_then_noise_only_for_full_training(mode, cheap, expected, monkeypatch):
    game = TicTacToe()
    state = np.ones((3, 3), dtype=np.int8)
    state.flat[1:3] = 0
    config = SearchConfig.from_args({
        "mode": mode, "root_policy_temperature_early": 2,
        "dirichlet_noise_weight": 0.5,
    }, 3)
    calls = []

    def noise(alpha):
        calls.append(alpha)
        return np.array([0.25, 0.75])

    monkeypatch.setattr(np.random, "dirichlet", noise)
    search = Search(game, state, 1, config, SearchBudget(1, cheap), 0)
    logits = np.zeros(9)
    logits[0] = 1000
    logits[1:3] = np.log([0.9, 0.1])
    search.complete(search.root, logits, 0.2)
    priors = np.zeros(9)
    for child in search.root.children:
        priors[child.action_taken] = child.prior
    np.testing.assert_allclose(priors[:3], expected)
    assert len(calls) == int(mode == "train" and not cheap)
    search.close()


@pytest.mark.parametrize("winner, player, expected", [
    (1, 1, [1, 0, 0]), (1, -1, [0, 0, 1]), (0, -1, [0, 1, 0]),
])
def test_wdl_target_order(winner, player, expected):
    np.testing.assert_array_equal(value_target(winner, player), expected)


def test_all_policy_heads_and_wdl_train_through_recurrence():
    torch.manual_seed(3)
    model = MuZeroNet(3, 3, num_channels=8)
    hidden = model.representation(torch.randn(2, 3, 3, 3))
    hidden = model.dynamics(hidden, torch.tensor([1, 7]))
    policy, value = model.prediction(hidden)
    assert policy.shape == (2, 4, 9)
    assert value.shape == (2, 3)
    loss = -policy.log_softmax(-1)[..., 0].sum() - value.log_softmax(-1)[:, 0].sum()
    loss.backward()
    gradients = model.prediction.policy_head[-1].weight.grad.flatten(1).abs().sum(1)
    assert (gradients > 0).all()
    for module in (model.representation, model.dynamics, model.prediction.value_head):
        assert sum(p.grad.abs().sum().item() for p in module.parameters()) > 0


def test_search_uses_only_main_policy_and_wdl_expectation():
    game = TicTacToe()
    model = MuZeroNet(3, 3, num_channels=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.prediction.value_head[-1].bias.copy_(torch.log(torch.tensor([0.6, 0.3, 0.1])))
        model.prediction.policy_head[-1].bias.copy_(torch.tensor([0., 100., -100., 20.]))
    mcts = MCTS(game, {"mode": "eval"}, model, "cpu")
    state = game.get_initial_state()
    state[0, 0] = 1
    result = mcts.search(state, -1, 0)
    policy, value = result.policy_target, result.root_value
    np.testing.assert_allclose(policy, [0] + [1 / 8] * 8)
    assert value == pytest.approx(0.5)


def test_replay_preserves_cheap_transition_and_uses_next_step_weight():
    records = trajectory()
    sample = ReplayBuffer()._build_sample(records, 0, 2, 9)
    np.testing.assert_array_equal(sample["actions"], [0, 3])
    np.testing.assert_array_equal(sample["policy_mask"], [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    for k in range(3):
        np.testing.assert_array_equal(sample["policy_targets"][k, PolicyHead.MAIN], records[k]["mcts_policy"])
        np.testing.assert_array_equal(sample["policy_targets"][k, PolicyHead.OPPONENT], records[k + 1]["mcts_policy"])
        np.testing.assert_array_equal(sample["value_targets"][k], records[k]["value_target"])
    soft = sample["policy_targets"][1, PolicyHead.SOFT]
    assert soft[0] == 0
    assert np.all(soft[records[1]["legal_actions"]] > 0)
    assert soft[3] < 1
    assert soft.sum() == pytest.approx(1)


def test_terminal_opponent_mask_and_wdl_absorption():
    sample = ReplayBuffer()._build_sample(trajectory(), 4, 3, 9)
    np.testing.assert_array_equal(sample["policy_mask"], [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(sample["value_targets"], [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
    records = trajectory()
    for record in records:
        record["value_target"] = value_target(0, record["player"])
    sample = ReplayBuffer()._build_sample(records, 4, 3, 9)
    np.testing.assert_array_equal(sample["value_targets"], [[0, 1, 0]] * 4)


def test_augmentation_preserves_four_head_layout_and_action_alignment(monkeypatch):
    sample = ReplayBuffer()._build_sample(trajectory(), 0, 2, 9)
    monkeypatch.setattr(np.random, "randint", lambda *args: 1)
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: False)
    augmented, = random_augment_batch([sample], 3)
    expected = np.rot90(sample["policy_targets"].reshape(3, 4, 3, 3), axes=(-2, -1))
    np.testing.assert_array_equal(augmented["policy_targets"], expected.reshape(3, 4, 9))
    assert augmented["actions"][0] == augmented["policy_targets"][0, PolicyHead.MAIN].argmax()
    np.testing.assert_array_equal(augmented["policy_mask"], sample["policy_mask"])


def test_move_temperature_does_not_modify_training_policy(monkeypatch):
    config = SearchConfig.from_args({"chosen_move_temperature_early": 2}, 3)
    history = GameHistory(TicTacToe(), config)
    policy = np.array([0.8, 0.2] + [0.] * 7)
    draws = []

    def choose(size, p):
        draws.append(p.copy())
        return 1

    monkeypatch.setattr(np.random, "choice", choose)
    history.play(SearchResult(policy, policy, 0.0), SearchBudget(8))
    np.testing.assert_allclose(draws[0][:2], [2 / 3, 1 / 3])
    np.testing.assert_array_equal(history.steps[0]["mcts_policy"], policy)
    assert history.steps[0]["action"] == 1


def test_backup_flips_values_without_changing_network_fpu_baseline():
    game = TicTacToe()
    config = SearchConfig.from_args({}, 3)
    search = Search(game, game.get_initial_state(), 1, config, SearchBudget(2), 0)
    search.complete(search.root, np.zeros(9), 0.8)
    child = search.next_leaf()
    search.complete(child, np.zeros(9), 0.4)
    assert search.root.nn_value == 0.8
    assert search.root.q_value() == pytest.approx(0.2)
    assert child.nn_value == 0.4
    assert child.q_value() == 0.4
    assert len(child.children) == 9
    search.close()


@pytest.mark.parametrize("args", [
    {"cheap_search_prob": -0.1}, {"cheap_search_prob": 1.1},
    {"num_simulations": 0}, {"num_simulations": 2.5}, {"cheap_search_visits": 0},
    {"chosen_move_temperature_halflife": 0}, {"root_policy_temperature": 0},
    {"fpu_reduction_max": -1}, {"pb_c_base": 0},
])
def test_rejects_invalid_search_configuration(args):
    with pytest.raises(ValueError):
        SearchConfig.from_args(args, 3)


def test_cheap_budget_never_exceeds_full_budget():
    config = SearchConfig.from_args({
        "num_simulations": 2, "cheap_search_visits": 20, "cheap_search_prob": 1,
    }, 3)
    assert config.sample_budget() == SearchBudget(2, True)
