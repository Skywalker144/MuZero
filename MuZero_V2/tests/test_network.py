from pathlib import Path
import tempfile
import unittest

import torch

from muzero.config import ROOT, load_config
from muzero.network import InferenceModule, ResBlock
from muzero.protocol import INPUT_PLANES, Plane
from muzero.replay import read_shard, Replay
from muzero.train import WeightAverage, create_model, create_optimizer, export_model, load_checkpoint, save_checkpoint, train_iteration
from test_learning import write_game


ARCHITECTURE = {
    'HIDDEN_STATE_NUM_CHANNELS': '7',
    'REPRESENTATION_NUM_BLOCKS': '1', 'REPRESENTATION_NUM_CHANNELS': '8',
    'DYNAMICS_NUM_BLOCKS': '2', 'DYNAMICS_NUM_CHANNELS': '10',
    'PREDICTION_BACKBONE_NUM_BLOCKS': '3', 'PREDICTION_BACKBONE_NUM_CHANNELS': '12',
    'POLICY_HEAD_NUM_BLOCKS': '1', 'POLICY_HEAD_NUM_CHANNELS': '6',
    'VALUE_HEAD_NUM_BLOCKS': '2', 'VALUE_HEAD_NUM_CHANNELS': '9',
    'VALUE_HEAD_HIDDEN_CHANNELS': '5',
}


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **overrides):
        return load_config(ROOT / 'configs/minimal_test', environ=ARCHITECTURE | {
            'BOARD_SIZES': '5', 'TRAIN_STEPS': '1',
        } | overrides)

    def test_independent_depths_and_widths(self):
        model = create_model(self.config())
        for module, blocks, channels in (
            (model.representation, 1, 8), (model.dynamics, 2, 10),
            (model.prediction.backbone, 3, 12),
            (model.prediction.policy_trunk, 1, 6),
            (model.prediction.value_trunk, 2, 9),
        ):
            residuals = [child for child in module.modules() if isinstance(child, ResBlock)]
            self.assertEqual(len(residuals), blocks)
            self.assertTrue(all(block.conv1.in_channels == channels for block in residuals))
        self.assertEqual(model.prediction.value_head[0].in_features, 9)
        self.assertEqual(model.prediction.value_head[0].out_features, 5)

    def test_export_recurrence_with_independent_widths_and_zero_blocks(self):
        for zero_blocks in (False, True):
            for value_head in ('wdl', 'scalar'):
                overrides = {key: '0' for key in ARCHITECTURE if key.endswith('_NUM_BLOCKS')} if zero_blocks else {}
                c = self.config(**overrides, VALUE_HEAD=value_head)
                model = create_model(c).eval()
                if zero_blocks:
                    self.assertFalse(any(isinstance(child, ResBlock) for child in model.modules()))
                with self.subTest(zero_blocks=zero_blocks, value_head=value_head), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'model.pt'
                    export_model(model, path)
                    scripted = torch.jit.load(str(path))
                    eager = InferenceModule(model)
                    for batch in (1, 3):
                        obs = torch.rand(batch, INPUT_PLANES, 5, 5)
                        obs[:, Plane.ON_BOARD] = 1
                        actual, expected = scripted.initial(obs), eager.initial(obs)
                        for _ in range(4):
                            self.assertEqual(actual[0].shape, (batch, 8, 5, 5))
                            torch.testing.assert_close(actual[0][:, -1], obs[:, Plane.ON_BOARD])
                            self.assertTrue((actual[0][:, :-1] >= 0).all())
                            self.assertTrue((actual[0][:, :-1] <= 1).all())
                            for a, e in zip(actual, expected):
                                torch.testing.assert_close(a, e)
                            actions = torch.arange(batch)
                            actual = scripted.recurrent(actual[0], actions)
                            expected = eager.recurrent(expected[0], actions)

    def test_padding_invariance_with_all_branches(self):
        small = create_model(self.config()).eval()
        large = create_model(self.config(BOARD_SIZES='9')).eval()
        large.load_state_dict(small.state_dict())
        obs = torch.rand(2, INPUT_PLANES, 5, 5)
        obs[:, Plane.ON_BOARD] = 1
        padded = torch.randn(2, INPUT_PLANES, 9, 9)
        padded[:, Plane.ON_BOARD] = 0
        padded[:, :, :5, :5] = obs
        a, b = InferenceModule(small).initial(obs), InferenceModule(large).initial(padded)
        for step in range(3):
            torch.testing.assert_close(a[0], b[0][:, :, :5, :5], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(a[1].reshape(2, 5, 5), b[1].reshape(2, 9, 9)[:, :5, :5], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(a[2], b[2], atol=2e-5, rtol=2e-5)
            self.assertTrue((b[0][:, :, 5:] == 0).all())
            self.assertTrue((b[1].softmax(-1).reshape(2, 9, 9)[:, 5:] == 0).all())
            a = InferenceModule(small).recurrent(a[0], torch.tensor([step * 5 + 1] * 2))
            b = InferenceModule(large).recurrent(b[0], torch.tensor([step * 9 + 1] * 2))

    def test_unrolled_training_and_checkpoint_architecture(self):
        c = self.config()
        model = create_model(c)
        optimizer = create_optimizer(model, c)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with tempfile.TemporaryDirectory() as tmp:
            game = Path(tmp) / 'game.mzs'
            write_game(game)
            replay = Replay([read_shard(game, 5)[0]], c)
            metrics = train_iteration(model, optimizer, replay, c, 0, 'cpu', WeightAverage(model, c))
            for name, parameter in model.named_parameters():
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertFalse(torch.equal(parameter, before[name]))
            checkpoint = Path(tmp) / 'checkpoint.pt'
            save_checkpoint(checkpoint, model, optimizer, c, 1, metrics, 1, WeightAverage(model, c))
            restored = create_model(c)
            load_checkpoint(checkpoint, restored, create_optimizer(restored, c), c, WeightAverage(restored, c))
            for name, parameter in restored.state_dict().items():
                torch.testing.assert_close(parameter, model.state_dict()[name])
            for key in ARCHITECTURE:
                changed = dict(c, **{key: c[key] + 1})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'mismatch'):
                    load_checkpoint(checkpoint, model, optimizer, changed, WeightAverage(model, changed))
