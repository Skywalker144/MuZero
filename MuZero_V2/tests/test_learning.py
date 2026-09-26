from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch

from muzero.config import ROOT, load_config
from muzero.network import InferenceModule
from muzero.protocol import INPUT_PLANES, MAGIC, VERSION, Plane, Rule
from muzero.replay import GameRecord, Replay, augment, window_size
from muzero.train import create_model, create_optimizer, export_model, train_iteration


def write_game(path, size=5, canvas=5, rule=Rule.FREESTYLE):
    board = np.zeros((canvas, canvas), dtype=np.int8)
    on_board = np.zeros_like(board, dtype=bool)
    on_board[:size, :size] = True
    with path.open('wb') as stream:
        stream.write(struct.pack('<8sIIIiI', MAGIC, canvas, size, int(rule), 1, 9))
        for turn, local in enumerate([0, size, 1, size + 1, 2, size + 2, 3, size + 3, 4]):
            action = local // size * canvas + local % size
            player = 1 if turn % 2 == 0 else -1
            observation = np.zeros((INPUT_PLANES, canvas, canvas), dtype=np.uint8)
            observation[Plane.OWN] = board == player
            observation[Plane.OPPONENT] = board == -player
            observation[Plane.BLACK_TO_MOVE] = on_board & (player == 1)
            observation[Plane.ON_BOARD] = on_board
            observation[Plane.STANDARD] = on_board & (rule == Rule.STANDARD)
            observation[Plane.RENJU] = on_board & (rule == Rule.RENJU)
            legal = (board == 0) & on_board
            policy = legal.astype(np.float32) / legal.sum()
            stream.write(struct.pack('<iIfI', player, action, float(turn % 2 == 0), 4))
            stream.write(observation.tobytes())
            stream.write(policy.astype('<f4').tobytes())
            board.flat[action] = player


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5'})
        torch.set_num_threads(config['TORCH_THREADS'])

    def test_export_dynamic_batches_and_recurrence(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5'})
        model = create_model(c).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            export_model(model, path)
            scripted = torch.jit.load(str(path))
            eager = InferenceModule(model)
            self.assertEqual(scripted.metadata(), (5, VERSION))
            for batch in (1, 4):
                observations = torch.rand(batch, INPUT_PLANES, 5, 5)
                observations[:, Plane.ON_BOARD] = 1
                actual = scripted.initial(observations)
                expected = eager.initial(observations)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e)
                actions = torch.arange(batch) % 25
                for a, e in zip(scripted.recurrent(actual[0], actions), eager.recurrent(expected[0], actions)):
                    torch.testing.assert_close(a, e)

    def test_unroll_mask_outcome_and_optimizer(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5'})
        c['SYMMETRY_AUGMENTATION'] = 0
        c['UNROLL_STEPS'] = 10
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzg'
            write_game(path)
            record = GameRecord.inspect(path, 5)
            replay = Replay([record], c)
            samples = replay.sample(np.random.default_rng(2))
            observations, actions, policies, values, masks = samples
            np.testing.assert_allclose(values.sum(-1), 1)
            self.assertTrue((masks[:, -1] == 0).all())
            self.assertTrue((policies[:, -1] == 0).all())
            for row in range(len(observations)):
                start = int(observations[row, :2].sum())
                for step in range(c['UNROLL_STEPS'] + 1):
                    current = start + step
                    main = float(current < 9 and current % 2 == 0)
                    opponent = float(current + 1 < 9 and (current + 1) % 2 == 0)
                    np.testing.assert_array_equal(masks[row, step], [main, main, opponent, opponent])
                for step in range(1, c['UNROLL_STEPS'] + 1):
                    np.testing.assert_array_equal(values[row, step], values[row, step - 1][::-1])
            model = create_model(c)
            before = next(model.parameters()).detach().clone()
            optimizer = create_optimizer(model, c)
            result = train_iteration(model, optimizer, replay, c, 0, 'cpu')
            self.assertTrue(np.isfinite(result['loss']))
            self.assertFalse(torch.equal(before, next(model.parameters())))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                GameRecord.inspect(path, 5)

    def test_symmetry_keeps_actions_and_policies_aligned(self):
        observations = np.zeros((8, 3, 5, 5), dtype=np.float32)
        observations[:, 0, 0, 1] = 1
        actions = np.ones((8, 2), dtype=np.int64)
        policies = np.zeros((8, 3, 4, 25), dtype=np.float32)
        policies[..., 1] = 1
        obs, transformed, targets = augment(observations, actions, policies, np.random.default_rng(0))
        np.testing.assert_array_equal(obs[:, 0].reshape(8, 25).argmax(-1), transformed[:, 0])
        np.testing.assert_array_equal(targets[:, 0, 0].argmax(-1), transformed[:, 0])

    def test_scalar_single_head_training_and_export(self):
        c = load_config(ROOT / 'configs/muzero', environ={
            'BOARD_SIZES': '5', 'BATCH_SIZE': '4', 'TRAIN_STEPS': '1', 'NUM_CHANNELS': '16',
        })
        model = create_model(c)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzg'
            write_game(path)
            replay = Replay([GameRecord.inspect(path, 5)], c)
            self.assertEqual(replay.sample(np.random.default_rng(0))[2].shape[2], 1)
            optimizer = create_optimizer(model, c)
            result = train_iteration(model, optimizer, replay, c, 0, 'cpu')
            self.assertTrue(np.isfinite(result['value_loss']))
            self.assertEqual(model.prediction.policy_head.out_channels, 1)
            self.assertEqual(model.prediction.value_head[-1].out_features, 1)
            exported = Path(tmp) / 'model.pt'
            export_model(model, exported)
            scripted = torch.jit.load(str(exported))
            observation = torch.rand(2, INPUT_PLANES, 5, 5)
            observation[:, Plane.ON_BOARD] = 1
            for actual, expected in zip(scripted.initial(observation), InferenceModule(model.eval()).initial(observation)):
                torch.testing.assert_close(actual, expected)
            logits = torch.tensor([[0.0], [1.0]])
            torch.testing.assert_close(model.prediction.utility(logits), logits[:, 0].tanh())
            self.assertEqual(window_size(100, c), c['FIXED_WINDOW_ROWS'])
            self.assertEqual(window_size(10000000, c), c['FIXED_WINDOW_ROWS'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
