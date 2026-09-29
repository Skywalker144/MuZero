from pathlib import Path
import tempfile
import unittest
import copy

import numpy as np
import torch
import torch.nn.functional as F

from muzero.config import ROOT, load_config, load_eval_config
from muzero.protocol import Plane
from muzero.replay import Replay, read_shard, PolicyHead
from muzero.train import create_model, WeightAverage, train_iteration
from test_learning import write_game


class AlignmentTests(unittest.TestCase):
    def test_reference_loss_and_inference_defaults(self):
        c = load_config(ROOT / 'configs/baseline', environ={})
        self.assertEqual(c['SOFT_POLICY_LOSS_SCALE'], 8.0)
        self.assertAlmostEqual(c['VALUE_LOSS_SCALE'], 1.2 * 0.6)
        self.assertEqual(c['NN_POLICY_TEMPERATURE'], 1.1)
        self.assertEqual(load_eval_config(ROOT / 'configs/baseline', environ={})['NN_POLICY_TEMPERATURE'], 1.1)
        original = load_config(ROOT / 'configs/exp_muzero', environ={})
        self.assertEqual(original['NN_POLICY_TEMPERATURE'], 1.0)
        self.assertEqual(original['ROOT_DESIRED_PER_CHILD_VISITS_COEFF'], 0)
        self.assertFalse(original['USE_POLICY_TARGET_PRUNING'])

    def test_sampling_and_soft_targets(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5'})
        c.update(BATCH_SIZE=8, UNROLL_STEPS=2, SYMMETRY_AUGMENTATION=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path)
            replay = Replay(read_shard(path, 5), c)
            for seed in range(8):
                obs, actions, policies, values, masks, weights = replay.sample(np.random.default_rng(seed))
                starts = obs[:, [Plane.OWN, Plane.OPPONENT]].sum((1, 2, 3)).astype(int)
                self.assertTrue((starts % 2 == 0).all())
                np.testing.assert_array_equal(weights[:, 0], 1)
                for i, start in enumerate(starts):
                    for step in range(3):
                        row = start + step
                        if row >= 9:
                            np.testing.assert_array_equal(masks[i, step], 1)
                            np.testing.assert_allclose(policies[i, step], 1 / 25)
                            continue
                        self.assertEqual(weights[i, step] > 0, row % 2 == 0)
                        np.testing.assert_array_equal(masks[i, step], 1)
                        for offset, head in [(0, PolicyHead.SOFT), (1, PolicyHead.SOFT_OPPONENT)]:
                            if row + offset >= 9:
                                continue
                            reference = (replay.games[0][row + offset]['policy'] + 1e-7) ** 0.25
                            reference /= reference.sum()
                            np.testing.assert_allclose(policies[i, step, head], reference, rtol=1e-6)

    def test_invalid_new_parameters(self):
        for overrides in [{'NN_POLICY_TEMPERATURE': '0'}, {'ROOT_DESIRED_PER_CHILD_VISITS_COEFF': '-1'},
                          {'POLICY_SURPRISE_DATA_WEIGHT': '0.8', 'VALUE_SURPRISE_DATA_WEIGHT': '0.3'}]:
            with self.assertRaises(ValueError):
                load_config(ROOT / 'configs/baseline', environ=overrides)

    def test_weighted_losses_and_gradients(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={
            'BOARD_SIZES': '5', 'UNROLL_STEPS': '0', 'TRAIN_STEPS': '1', 'BATCH_SIZE': '4',
        })
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path)
            replay = Replay(read_shard(path, 5), c)
            batch = replay.sample(np.random.default_rng(7))
            batch[-1][:, 0] = [0, 0.25, 0.5, 1]

            class FixedBatch:
                def sample(self, rng):
                    return batch

            model = create_model(c)
            expected = copy.deepcopy(model)
            observations, _, policies, values, masks, weights = map(torch.from_numpy, batch)
            logits, value_logits = expected.prediction(expected.representation(observations))
            loss = torch.zeros(())
            head_losses = {}
            for head, scale in enumerate([1, 8, 0.15, 1.2]):
                rows = F.cross_entropy(logits[:, head], policies[:, 0, head], reduction='none')
                head_loss = (rows * masks[:, 0, head] * weights[:, 0]).mean() * scale
                head_losses[PolicyHead(head).name.lower()] = head_loss.item()
                loss = loss + head_loss
            loss = loss + 0.72 * (F.cross_entropy(value_logits, values[:, 0], reduction='none') * weights[:, 0]).mean()
            loss.backward()
            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            metrics = train_iteration(model, optimizer, FixedBatch(), c, 0, 'cpu', WeightAverage(model, c))
            self.assertAlmostEqual(metrics['loss'], loss.item(), places=5)
            self.assertEqual(set(metrics['policy_head_losses']), set(head_losses))
            for name, value in head_losses.items():
                self.assertAlmostEqual(metrics['policy_head_losses'][name], value, places=5)
            self.assertAlmostEqual(sum(metrics['policy_head_losses'].values()), metrics['policy_loss'], places=5)
            for actual, reference in zip(model.parameters(), expected.parameters()):
                if reference.grad is not None:
                    torch.testing.assert_close(actual.grad, reference.grad, rtol=2e-5, atol=1e-6)

    def test_fractional_weights_and_empty_training_mass(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5', 'BATCH_SIZE': '1'})
        c['SYMMETRY_AUGMENTATION'] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path, weights=[0.25, 0.75] + [0] * 7)
            record = read_shard(path, 5)[0]
            replay = Replay([record], c)
            rng = np.random.default_rng(9)
            starts = [int(replay.sample(rng)[0][0, [Plane.OWN, Plane.OPPONENT]].sum()) for _ in range(512)]
            self.assertEqual(set(starts), {0, 1})
            self.assertAlmostEqual(np.mean(starts), 0.75, delta=0.07)
            write_game(path, weights=[0] * 9)
            replay = Replay(read_shard(path, 5), c)
            with self.assertRaisesRegex(ValueError, 'no positive'):
                replay.sample(np.random.default_rng(1))
