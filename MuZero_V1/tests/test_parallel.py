import numpy as np
import pytest
import torch

from muzero import MuZero, ParallelSelfPlayer
from muzero.mcts import MCTS, Node
from muzero.targets import value_target
from envs.gomoku import Gomoku
from envs.tictactoe import TicTacToe


def make_args(**overrides):
    args = {
        "num_simulations": 8,
        "pb_c_init": 1.25,
        "pb_c_base": 19652,
        "num_blocks": 1,
        "num_channels": 8,
        "dirichlet_total_concentration": 0.03 * 3 ** 2,
        "dirichlet_noise_weight": 0.25,
        "num_parallel_games": 4,
    }
    args.update(overrides)
    return args


def make_model(game):
    from muzero.network import MuZeroNet

    return MuZeroNet(
        game.board_size, game.num_planes, num_blocks=1, num_channels=8
    ).eval()


def make_player(game=None, args=None):
    game = game or TicTacToe()
    return game, ParallelSelfPlayer(game, args or make_args(), make_model(game), "cpu")


def assert_valid_game(samples, winner, game_len, game):
    assert game_len > 0
    assert winner in (-1, 0, 1)
    for sample in samples:
        assert sample["observation"].shape == (
            game.num_planes, game.board_size, game.board_size
        )
        assert sample["player"] in (-1, 1)
        assert 0 <= sample["action"] < game.board_size ** 2
        assert sample["mcts_policy"].shape == (game.board_size ** 2,)
        assert np.isclose(sample["mcts_policy"].sum(), 1.0)
        np.testing.assert_array_equal(sample["value_target"], value_target(winner, sample["player"]))


class TestBatchedInference:
    def test_matches_sequential_inference(self):
        game = Gomoku(board_size=9)
        args = make_args(dirichlet_total_concentration=0.03 * 9 ** 2)
        model = make_model(game)
        mcts = MCTS(game, args, model, "cpu")
        player = ParallelSelfPlayer(game, args, model, "cpu")

        state = game.get_initial_state()
        for action, to_play in [(40, 1), (30, -1), (41, 1)]:
            state = game.get_next_state(state, action, to_play)

        # 根节点：h + f 与串行一致
        root = Node(1, state=state)
        (root_logits, root_value), = player.mcts.infer_nodes([root])
        with torch.inference_mode():
            hidden = model.representation(torch.tensor(game.encode_state(state, 1), dtype=torch.float32).unsqueeze(0))
            policy_logits, wdl_logits = model.prediction(hidden)
            seq_logits = policy_logits[0, 0].numpy()
            wdl = wdl_logits.softmax(-1)[0]
            seq_value = (wdl[0] - wdl[2]).item()
        assert np.allclose(root_logits, seq_logits, atol=1e-5)
        assert np.isclose(root_value, seq_value, atol=1e-5)

        # 树节点：g + f 与串行一致
        child = Node(-1, prior=0.5, parent=root, action_taken=17)
        (child_logits, child_value), = player.mcts.infer_nodes([child])
        with torch.inference_mode():
            next_hidden = model.dynamics(hidden, torch.tensor([17]))
            policy_logits, wdl_logits = model.prediction(next_hidden)
            seq_logits = policy_logits[0, 0].numpy()
            wdl = wdl_logits.softmax(-1)[0]
            seq_value = (wdl[0] - wdl[2]).item()
        assert np.allclose(child_logits, seq_logits, atol=1e-5)
        assert np.isclose(child_value, seq_value, atol=1e-5)


class TestGames:
    def test_collects_requested_number_of_valid_games(self):
        game, player = make_player()
        games = list(player.run(6))
        assert len(games) == 6
        for samples, winner, game_len in games:
            assert_valid_game(samples, winner, game_len, game)

    def test_pool_shrinks_when_fewer_games_than_workers(self):
        game, player = make_player(args=make_args(num_parallel_games=8))
        games = list(player.run(2))
        assert len(games) == 2

    def test_batches_across_games(self):
        game, player = make_player(args=make_args(num_parallel_games=4))
        sizes = []
        original = player.mcts.infer_nodes

        def recording(nodes):
            sizes.append(len(nodes))
            return original(nodes)

        player.mcts.infer_nodes = recording
        list(player.run(8))
        assert max(sizes) > 1

    def test_simulation_budget_per_move(self):
        # 每步棋的 NN 评估次数不能超过 1 次根评估 + num_simulations。
        sims = 6
        game, player = make_player(args=make_args(num_simulations=sims, num_parallel_games=3))
        evals = [0]
        original = player.mcts.infer_nodes

        def counting(nodes):
            evals[0] += len(nodes)
            return original(nodes)

        player.mcts.infer_nodes = counting
        games = list(player.run(5))
        total_moves = sum(game_len for _, _, game_len in games)
        assert total_moves > 0
        expected = sum(step["search_visits"] + 1 for samples, _, _ in games for step in samples)
        assert evals[0] == expected
        assert evals[0] <= (sims + 1) * total_moves

    def test_deterministic_with_fixed_seed(self):
        game, player = make_player()
        np.random.seed(123)
        first = list(player.run(4))
        np.random.seed(123)
        second = list(player.run(4))

        assert len(first) == len(second)
        for (s1, w1, l1), (s2, w2, l2) in zip(first, second):
            assert (w1, l1) == (w2, l2)
            for a, b in zip(s1, s2):
                assert np.array_equal(a["observation"], b["observation"])
                assert np.array_equal(a["mcts_policy"], b["mcts_policy"])
                assert a["action"] == b["action"]
                np.testing.assert_array_equal(a["value_target"], b["value_target"])


class TestEquivalence:
    @pytest.mark.parametrize("pb_c_base, pb_c_init", [(19652, 1.25), (10, 0.5)])
    @pytest.mark.parametrize("cheap_prob", [0.0, 0.75, 1.0])
    def test_single_game_matches_sequential_backend(self, pb_c_base, pb_c_init, cheap_prob):
        # num_parallel_games=1 时，并行后端每轮 batch=1，推理结果与串行版
        # 逐位一致，且 RNG 消耗顺序也相同，因此应当产生完全相同的对局。
        game = TicTacToe()
        base_args = {
            "num_simulations": 12,
            "cheap_search_prob": cheap_prob,
            "cheap_search_visits": 3,
            "pb_c_init": pb_c_init,
            "pb_c_base": pb_c_base,
            "num_blocks": 1,
            "num_channels": 8,
            "dirichlet_total_concentration": 0.03 * 3 ** 2,
            "dirichlet_noise_weight": 0.25,
        }
        seq = MuZero(game, {**base_args, "parallel": False})
        par = MuZero(game, {**base_args, "parallel": True, "num_parallel_games": 1})
        par.model.load_state_dict(seq.model.state_dict())

        np.random.seed(42)
        seq_samples, seq_winner, seq_len = seq.selfplay()
        np.random.seed(42)
        par_samples, par_winner, par_len = next(par.parallel_player.run(1))

        assert (seq_winner, seq_len) == (par_winner, par_len)
        for a, b in zip(seq_samples, par_samples):
            assert np.array_equal(a["observation"], b["observation"])
            assert np.array_equal(a["mcts_policy"], b["mcts_policy"])
            assert a["action"] == b["action"]
            assert a["policy_weight"] == b["policy_weight"]
            assert a["search_visits"] == b["search_visits"]
            np.testing.assert_array_equal(a["value_target"], b["value_target"])


class TestTrainerIntegration:
    @pytest.fixture
    def tiny_args(self, tmp_path):
        return {
            "num_simulations": 8,
            "pb_c_init": 1.25,
            "pb_c_base": 19652,
            "num_blocks": 1,
            "num_channels": 8,
            "dirichlet_total_concentration": 0.03 * 3 ** 2,
            "dirichlet_noise_weight": 0.25,
            "unroll_steps": 3,
            "parallel": True,
            "num_parallel_games": 3,
            "num_iterations": 1,
            "train_steps": 2,
            "batch_size": 16,
            "replay_ratio": 8,
            "bootstrap_games": 2,
            "min_rows": 16,
            "taper_window_exponent": 1.0,
            "expand_window_per_row": 0.3,
            "save_interval": 1,
            "data_dir": str(tmp_path),
        }

    def test_run_iteration_collects_games(self, tiny_args):
        mz = MuZero(TicTacToe(), tiny_args)
        assert mz.parallel_player is not None
        mz._run_iteration(0)
        assert mz.game_count > 0
        assert len(mz.replay_buffer) > 0

    def test_learn_end_to_end(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2, "min_rows": 8}
        mz = MuZero(TicTacToe(), args)
        mz.learn()
        assert mz.iteration == 2
        assert mz.game_count > 0
        assert len(mz.metrics.game_records) == mz.game_count

    def test_sequential_path_unused_when_parallel(self, tiny_args, monkeypatch):
        mz = MuZero(TicTacToe(), tiny_args)

        def fail(*args, **kwargs):
            raise AssertionError("serial selfplay should not be called when parallel=True")

        monkeypatch.setattr(mz, "selfplay", fail)
        mz._run_iteration(0)
        assert mz.game_count > 0
