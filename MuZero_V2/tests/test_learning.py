from pathlib import Path
import os
import struct
import io
import zlib
import tempfile
import unittest

import numpy as np
import torch

from muzero.config import ROOT, load_config
from muzero.network import InferenceModule
from muzero.protocol import INPUT_PLANES, MAGIC, VERSION, Plane, Rule
from muzero.replay import read_shard, Replay, augment, window_size
from muzero.train import WeightAverage, create_model, create_optimizer, export_model, load_checkpoint, save_checkpoint, train_iteration


def write_game(path, size=5, canvas=5, rule=Rule.FREESTYLE, opening_count=0):
    board = np.zeros((canvas, canvas), dtype=np.int8)
    on_board = np.zeros_like(board, dtype=bool)
    on_board[:size, :size] = True
    opening = []
    with io.BytesIO() as stream:
        for turn, local in enumerate([0, size, 1, size + 1, 2, size + 2, 3, size + 3, 4]):
            action = local // size * canvas + local % size
            player = 1 if turn % 2 == 0 else -1
            if turn < opening_count:
                opening.append(action)
                board.flat[action] = player
                continue
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
            stream.write(np.packbits(observation.reshape(-1)).tobytes())
            stream.write(policy.astype('<f4').tobytes())
            board.flat[action] = player
        compressed = zlib.compress(stream.getvalue())
    path.write_bytes(struct.pack('<8sI', MAGIC, 1) + struct.pack('<IIIIiIII', 0, canvas, size, int(rule), 1,
                     9 - opening_count, len(opening), len(compressed)) +
                     struct.pack(f'<{len(opening)}I', *opening) + compressed)


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

    @unittest.skipUnless(os.environ.get('MUZERO_TEST_DEVICE', 'cpu').startswith('cuda'), 'Set MUZERO_TEST_DEVICE=cuda:0')
    def test_cuda_checkpoint_restores_optimizer_counters_on_cpu(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5'})
        model = create_model(c)
        optimizer = create_optimizer(model, c)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            save_checkpoint(path, model, optimizer, c, 0, {}, 1, WeightAverage(model, c))
            device = os.environ['MUZERO_TEST_DEVICE']
            restored = create_model(c).to(device)
            restored_optimizer = create_optimizer(restored, c)
            load_checkpoint(path, restored, restored_optimizer, c, WeightAverage(restored, c))
            for state in restored_optimizer.state.values():
                self.assertEqual(state['step'].device.type, 'cpu')
                self.assertEqual(state['step'].item(), 1)
                self.assertEqual(state['exp_avg'].device.type, 'cuda')
            for parameter in restored.parameters():
                parameter.grad = torch.ones_like(parameter)
            restored_optimizer.step()
            optimizer.step()
            for actual, expected in zip(restored.parameters(), model.parameters()):
                torch.testing.assert_close(actual.cpu(), expected)

    def test_nonfinite_gradient_does_not_update_parameters(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5', 'TRAIN_STEPS': '1'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path)
            replay = Replay([read_shard(path, 5)[0]], c)
            model = create_model(c)
            optimizer = create_optimizer(model, c)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            handle = next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
            with self.assertRaisesRegex(FloatingPointError, 'gradient'):
                train_iteration(model, optimizer, replay, c, 0, 'cpu', WeightAverage(model, c))
            handle.remove()
            self.assertEqual(len(optimizer.state), 0)
            for actual, expected in zip(model.parameters(), before):
                torch.testing.assert_close(actual, expected)

    def test_unroll_mask_outcome_and_optimizer(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5'})
        c['SYMMETRY_AUGMENTATION'] = 0
        c['UNROLL_STEPS'] = 10
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path)
            record = read_shard(path, 5)[0]
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
            result = train_iteration(model, optimizer, replay, c, 0, 'cpu', WeightAverage(model, c))
            self.assertTrue(np.isfinite(result['loss']))
            self.assertFalse(torch.equal(before, next(model.parameters())))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                read_shard(path, 5)[0]

    def test_symmetry_keeps_actions_and_policies_aligned(self):
        observations = np.zeros((8, 3, 5, 5), dtype=np.float32)
        observations[:, 0, 0, 1] = 1
        actions = np.ones((8, 2), dtype=np.int64)
        policies = np.zeros((8, 3, 4, 25), dtype=np.float32)
        policies[..., 1] = 1
        obs, transformed, targets = augment(observations, actions, policies, np.random.default_rng(0))
        np.testing.assert_array_equal(obs[:, 0].reshape(8, 25).argmax(-1), transformed[:, 0])
        np.testing.assert_array_equal(targets[:, 0, 0].argmax(-1), transformed[:, 0])

    def test_plot_diagnostics_match_loss_and_module_gradients(self):
        for unroll in (0, 2):
            with self.subTest(unroll=unroll), tempfile.TemporaryDirectory() as tmp:
                c = load_config(ROOT / 'configs/minimal_test', environ={
                    'BOARD_SIZES': '5', 'TRAIN_STEPS': '1', 'UNROLL_STEPS': str(unroll),
                    'VALUE_LOSS_SCALE': '0.7',
                })
                path = Path(tmp) / 'game.mzs'
                write_game(path)
                replay = Replay([read_shard(path, 5)[0]], c)
                model = create_model(c)
                result = train_iteration(model, create_optimizer(model, c), replay, c, 0, 'cpu', WeightAverage(model, c))
                self.assertEqual(len(result['step_losses']), unroll + 1)
                self.assertAlmostEqual(sum(result['step_losses']), result['loss'], places=5)
                for name in ('representation', 'dynamics', 'prediction'):
                    squared = sum(float(parameter.grad.square().sum())
                                  for parameter in getattr(model, name).parameters() if parameter.grad is not None)
                    self.assertAlmostEqual(result['grad_norms'][name], squared ** .5, places=5)

    def test_scalar_single_head_training_and_export(self):
        c = load_config(ROOT / 'configs/muzero', environ={
            'BOARD_SIZES': '5', 'BOARD_SIZE_WEIGHTS': '1', 'BATCH_SIZE': '4', 'TRAIN_STEPS': '1',
            'HIDDEN_STATE_NUM_CHANNELS': '16', 'REPRESENTATION_NUM_CHANNELS': '16',
            'DYNAMICS_NUM_CHANNELS': '16', 'PREDICTION_BACKBONE_NUM_CHANNELS': '16',
            'POLICY_HEAD_NUM_CHANNELS': '16', 'VALUE_HEAD_NUM_CHANNELS': '16',
            'VALUE_HEAD_HIDDEN_CHANNELS': '8', 'REPLAY_WINDOW': 'fixed',
        })
        model = create_model(c)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path)
            replay = Replay([read_shard(path, 5)[0]], c)
            self.assertEqual(replay.sample(np.random.default_rng(0))[2].shape[2], 1)
            optimizer = create_optimizer(model, c)
            result = train_iteration(model, optimizer, replay, c, 0, 'cpu', WeightAverage(model, c))
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
