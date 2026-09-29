from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from muzero.config import ROOT, load_config
from muzero.protocol import Plane
from muzero.replay import Outcome, PolicyHead, Replay, read_shard
from muzero.train import WeightAverage, create_model, create_optimizer, train_iteration
from test_learning import write_game


class AbsorbingTests(unittest.TestCase):
    def config(self, **overrides):
        return load_config(ROOT / 'configs/minimal_test', environ={
            'BOARD_SIZES': '5,7', 'BOARD_SIZE_WEIGHTS': '1,1', 'BATCH_SIZE': '4',
            'UNROLL_STEPS': '5', 'TRAIN_STEPS': '1', 'SYMMETRY_AUGMENTATION': 'false',
        } | overrides)

    def test_terminal_boundary_for_single_and_auxiliary_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path, size=5, canvas=7, weights=[0] * 8 + [1])
            record = read_shard(path, 7)[0]
            last = record.load()[-1]
            on_board = last['observation'][Plane.ON_BOARD].reshape(-1) != 0
            uniform = on_board.astype(np.float32) / on_board.sum()
            for auxiliary in (False, True):
                for unroll in (0, 5):
                    for winner in (-1, 0, 1):
                        with self.subTest(auxiliary=auxiliary, unroll=unroll, winner=winner):
                            c = self.config(AUXILIARY_POLICY_HEADS=str(auxiliary).lower(),
                                            UNROLL_STEPS=str(unroll), SOFT_POLICY_LOSS_SCALE='0',
                                            OPPONENT_POLICY_LOSS_SCALE='0')
                            replay = Replay([replace(record, winner=winner)], c)
                            obs, actions, policies, values, masks, weights = replay.sample(np.random.default_rng(7))
                            np.testing.assert_array_equal(obs, np.broadcast_to(last['observation'], obs.shape))
                            np.testing.assert_array_equal(masks, 1)
                            np.testing.assert_array_equal(weights, 1)
                            np.testing.assert_array_equal(policies[:, 0, PolicyHead.MAIN],
                                                          np.broadcast_to(last['policy'], (4, 49)))
                            if unroll:
                                np.testing.assert_array_equal(actions[:, 0], last['action'])
                                np.testing.assert_allclose(policies[:, 1:],
                                                           np.broadcast_to(uniform, policies[:, 1:].shape))
                            if auxiliary:
                                np.testing.assert_allclose(policies[:, 0, PolicyHead.OPPONENT:],
                                                           np.broadcast_to(uniform, (4, 2, 49)))
                                softened = np.zeros(49, dtype=np.float32)
                                softened[on_board] = (last['policy'][on_board] + c['SOFT_POLICY_EPS']) ** (1 / c['SOFT_POLICY_TEMPERATURE'])
                                softened /= softened.sum()
                                np.testing.assert_allclose(policies[:, 0, PolicyHead.SOFT],
                                                           np.broadcast_to(softened, (4, 49)))
                            for step in range(unroll + 1):
                                outcome = winner * (-1 if step % 2 else 1)
                                expected = np.zeros(3)
                                expected[Outcome.WIN if outcome > 0 else Outcome.LOSS if outcome < 0 else Outcome.DRAW] = 1
                                np.testing.assert_array_equal(values[:, step], np.broadcast_to(expected, (4, 3)))

    def test_random_actions_cover_board_and_are_reproducible(self):
        c = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path, size=5, canvas=7, weights=[0] * 8 + [1])
            replay = Replay(read_shard(path, 7), c)
            first = replay.sample(np.random.default_rng(3))
            second = replay.sample(np.random.default_rng(3))
            for actual, expected in zip(first, second):
                np.testing.assert_array_equal(actual, expected)
            rng = np.random.default_rng(19)
            actions = np.concatenate([replay.sample(rng)[1][:, 1:] for _ in range(64)])
            board_actions = {y * 7 + x for y in range(5) for x in range(5)}
            self.assertEqual(set(actions.ravel()), board_actions)
            for step in range(actions.shape[1]):
                self.assertEqual(set(actions[:, step]), board_actions)
            self.assertTrue((actions[:, 1:] != actions[:, :-1]).any())
            self.assertTrue((actions[:, 1:] == actions[:, :-1]).any())

    def test_symmetry_and_consistency_keep_terminal_boundaries(self):
        c = self.config(SYMMETRY_AUGMENTATION='true', USE_CONSISTENCY_LOSS='true', BATCH_SIZE='2')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path, size=5, canvas=7, opening_count=7, weights=[1] * 9)
            replay = Replay(read_shard(path, 7), c)
            for seed in range(16):
                obs, actions, policies, values, masks, weights, future, valid = replay.sample(np.random.default_rng(seed))
                np.testing.assert_array_equal(masks, 1)
                for i in range(len(obs)):
                    start = int(obs[i, [Plane.OWN, Plane.OPPONENT]].sum())
                    terminal = 9 - start
                    on_board = obs[i, Plane.ON_BOARD].reshape(-1)
                    uniform = on_board / on_board.sum()
                    self.assertTrue(on_board[actions[i]].all())
                    np.testing.assert_allclose(policies[i, terminal:],
                                               np.broadcast_to(uniform, policies[i, terminal:].shape))
                    np.testing.assert_allclose(policies[i, terminal - 1, PolicyHead.OPPONENT:],
                                               np.broadcast_to(uniform, (2, 49)))
                    np.testing.assert_array_equal(valid[i], np.arange(1, 6) < terminal)
                    self.assertFalse(future[i, terminal - 1:].any())
                    np.testing.assert_array_equal(weights[i, terminal:], 1)

    def test_terminal_policy_gradients_reach_every_head(self):
        torch.set_num_threads(1)
        c = self.config(UNROLL_STEPS='3')
        device = os.environ.get('MUZERO_TEST_DEVICE', 'cpu')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path, size=5, canvas=7, weights=[0] * 8 + [1])
            replay = Replay(read_shard(path, 7), c)
            batch = replay.sample(np.random.default_rng(c['SEED'] + 1))
            model = create_model(c).to(device)
            logits_by_step = []

            def capture(module, inputs, outputs):
                outputs[0].retain_grad()
                logits_by_step.append(outputs[0])

            handle = model.prediction.register_forward_hook(capture)
            try:
                metrics = train_iteration(model, create_optimizer(model, c), replay, c, 0,
                                          device, WeightAverage(model, c))
            finally:
                handle.remove()
            scales = torch.tensor([1, c['SOFT_POLICY_LOSS_SCALE'], c['OPPONENT_POLICY_LOSS_SCALE'],
                                   c['SOFT_POLICY_LOSS_SCALE'] * c['OPPONENT_POLICY_LOSS_SCALE']], device=device)
            self.assertEqual(len(logits_by_step), 4)
            for step, logits in enumerate(logits_by_step[1:], 1):
                targets = torch.from_numpy(batch[2][:, step]).to(device)
                on_board = torch.from_numpy(batch[0][:, Plane.ON_BOARD].reshape(4, 49)).to(device)
                uniform = on_board / on_board.sum(-1, keepdim=True)
                torch.testing.assert_close(targets, uniform[:, None].expand_as(targets))
                expected = (logits.detach().softmax(-1) - targets) * scales[None, :, None] / (4 * 3)
                torch.testing.assert_close(logits.grad, expected, atol=1e-7, rtol=1e-5)
                self.assertTrue((logits.grad.abs().sum((0, 2)) > 0).all())
            self.assertGreater(metrics['grad_norms']['dynamics'], 0)
            self.assertTrue(np.isfinite(metrics['loss']))
