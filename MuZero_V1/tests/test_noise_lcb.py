from dataclasses import replace

import numpy as np
import pytest

from envs.tictactoe import TicTacToe
from muzero.config import SearchBudget, SearchConfig
from muzero.mcts import Node, Search, SearchResult
from muzero.replay_buffer import ReplayBuffer
from muzero.selfplay import GameHistory
from muzero.utils import add_dirichlet_noise


def capture_noise(monkeypatch, policy, board_size, shaped=True):
    captured = []

    def sample(alpha):
        captured.append(np.array(alpha))
        return np.full(len(alpha), 1 / len(alpha))

    monkeypatch.setattr(np.random, "dirichlet", sample)
    legal = np.arange(len(policy)) < len(policy) - 1
    result = add_dirichlet_noise(
        np.array(policy), 3.0, legal, board_size=board_size, noise_weight=0.25, shaped=shaped,
    )
    return captured[0], result


def test_shaped_noise_keeps_uniform_mass_and_prefers_informative_priors(monkeypatch):
    alpha, result = capture_noise(monkeypatch, [0.8, 0.19, 0.01, 0.0], 9)
    assert alpha.sum() == pytest.approx(3)
    assert np.all(alpha >= 0.5)
    assert alpha[0] == pytest.approx(alpha[1])
    assert alpha[1] > alpha[2]
    np.testing.assert_allclose(result, [0.8 * 0.75 + 0.25 / 3, 0.19 * 0.75 + 0.25 / 3, 0.01 * 0.75 + 0.25 / 3, 0])


def test_shaped_noise_cap_scales_for_small_boards(monkeypatch):
    alpha, _ = capture_noise(monkeypatch, [0.8, 0.15, 0.05, 0.0], 3)
    assert alpha[0] > alpha[1] > alpha[2]


@pytest.mark.parametrize("shaped", [False, True])
def test_uniform_prior_remains_uniform_noise(shaped, monkeypatch):
    alpha, _ = capture_noise(monkeypatch, [1 / 3, 1 / 3, 1 / 3, 0.0], 3, shaped)
    np.testing.assert_allclose(alpha, [1, 1, 1])


def test_shaped_noise_handles_zero_prior_on_legal_move(monkeypatch):
    alpha, result = capture_noise(monkeypatch, [1.0, 0.0, 0.0, 0.0], 3)
    assert np.isfinite(alpha).all()
    assert np.all(alpha > 0)
    assert result[1] > 0 and result[2] > 0 and result[3] == 0


def prepared_search(mode="train", cheap=False, use_lcb=True, specs=None):
    game = TicTacToe()
    config = SearchConfig.from_args({"mode": mode, "use_lcb_for_selection": use_lcb}, 3)
    search = Search(game, game.get_initial_state(), 1, config, SearchBudget(120, cheap), 0)
    specs = specs or [(0, 20, -0.7), (1, 100, -0.2)]
    for action, visits, child_value in specs:
        node = Node(-1, prior=1 / len(specs), parent=search.root, action_taken=action)
        for _ in range(visits):
            node.update(child_value)
        search.root.children.append(node)
    search.root.update(0.0)
    return search


def test_lcb_promotes_reliably_better_move_even_at_index_zero():
    search = prepared_search()
    result = search.result()
    np.testing.assert_allclose(result.visit_policy[:2], [1 / 6, 5 / 6])
    assert result.policy_target.argmax() == 0
    assert result.policy_target[0] > result.visit_policy[0]
    assert np.all(result.policy_target[2:] == 0)
    assert result.policy_target.sum() == pytest.approx(1)
    assert [node.visits for node in search.root.children] == [20, 100]
    search.close()


def test_lcb_rejects_low_visit_candidate_even_with_high_value():
    search = prepared_search(specs=[(0, 14, -1.0), (1, 100, 0.0)])
    result = search.result()
    np.testing.assert_array_equal(result.policy_target, result.visit_policy)
    search.close()


@pytest.mark.parametrize("cheap, enabled", [(True, True), (False, False)])
def test_cheap_and_disabled_lcb_keep_raw_targets(cheap, enabled):
    search = prepared_search(cheap=cheap, use_lcb=enabled)
    result = search.result()
    np.testing.assert_array_equal(result.policy_target, result.visit_policy)
    search.close()


def test_one_visit_lcb_is_finite_and_does_not_visit_other_actions():
    search = prepared_search(specs=[(0, 1, -1), (1, 0, 0)])
    result = search.result()
    assert np.isfinite(result.policy_target).all()
    np.testing.assert_array_equal(result.policy_target, [1] + [0] * 8)
    search.close()


def test_equal_lcb_and_constant_values_are_finite():
    search = prepared_search(specs=[(0, 100, -0.5), (1, 100, -0.5)])
    result = search.result()
    np.testing.assert_allclose(result.policy_target[:2], [0.5, 0.5])
    search.close()


def test_return_variance_changes_lcb_target():
    stable = prepared_search(specs=[(0, 20, -0.3), (1, 100, -0.2)])
    variable = prepared_search(specs=[(0, 0, 0), (1, 100, -0.2)])
    for value in [-1.0, 0.4] * 10:
        variable.root.children[0].update(value)
    assert stable.result().policy_target.argmax() == 0
    assert variable.result().policy_target.argmax() == 1
    stable.close()
    variable.close()


def test_square_sum_tracks_actual_backups_and_sign_changes():
    search = prepared_search(specs=[(0, 0, 0)])
    child = search.root.children[0]
    search.complete(child, np.zeros(9), 0.6)
    assert child.value_sq_sum == pytest.approx(0.36)
    assert search.root.value_sq_sum == pytest.approx(0.36)
    assert child.value_sum == pytest.approx(0.6)
    assert search.root.value_sum == pytest.approx(-0.6)
    search.close()


def test_training_samples_raw_visits_but_replay_learns_lcb_targets(monkeypatch):
    search = prepared_search()
    history = GameHistory(search.game, replace(search.config, chosen_move_temperature_early=1))
    result = search.result()
    drawn = []

    def choose(count, p):
        drawn.append(p.copy())
        return 1

    monkeypatch.setattr(np.random, "choice", choose)
    history.play(result, search.budget)
    np.testing.assert_allclose(drawn[0], result.visit_policy)
    np.testing.assert_array_equal(history.steps[0]["mcts_policy"], result.policy_target)
    sample = ReplayBuffer()._build_sample(history.steps, 0, 0, 9)
    np.testing.assert_allclose(sample["policy_targets"][0, 0], result.policy_target)
    assert history.steps[0]["action"] == 1
    search.close()


def test_eval_moves_use_lcb_target():
    search = prepared_search(mode="eval")
    history = GameHistory(search.game, search.config)
    history.play(search.result(), search.budget)
    assert history.steps[0]["action"] == 0
    search.close()


@pytest.mark.parametrize("args", [{"lcb_stdevs": 0}, {"lcb_stdevs": -1}, {"min_visit_prop_for_lcb": 1.1}])
def test_invalid_lcb_configuration(args):
    with pytest.raises(ValueError):
        SearchConfig.from_args(args, 3)
