import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from muzero.config import CONFIG_FILES, KEYS, ROOT, load_config, model_identity, render_config
from muzero.protocol import Plane
from muzero.replay import Replay, read_shard
from muzero.run import initialize
from muzero.storage import checkpoint_path
from muzero.train import (WeightAverage, create_model, create_optimizer, export_model,
                          load_checkpoint, save_checkpoint, train_iteration)
from test_learning import write_game


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **overrides):
        return load_config(ROOT / 'configs/minimal_test', environ={
            'BOARD_SIZES': '5', 'TRAIN_STEPS': '1', 'USE_CONSISTENCY_LOSS': 'true',
        } | overrides)

    def test_profile_and_legacy_config(self):
        baseline = load_config(ROOT / 'configs/baseline', environ={})
        enabled = load_config(ROOT / 'configs/exp_consistency', environ={})
        self.assertFalse(baseline['USE_CONSISTENCY_LOSS'])
        self.assertTrue(enabled['USE_CONSISTENCY_LOSS'])
        self.assertEqual(model_identity(baseline), model_identity(enabled))
        self.assertEqual({key for key in baseline if baseline[key] != enabled[key]},
                         {'USE_CONSISTENCY_LOSS', 'DATA_DIR'})
        for overrides in ({'UNROLL_STEPS': '0'}, {'CONSISTENCY_LOSS_SCALE': '-1'},
                          {'CONSISTENCY_LOSS_SCALE': 'nan'}):
            with self.assertRaises(ValueError):
                self.config(**overrides)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for filename in CONFIG_FILES:
                text = render_config(baseline, filename)
                text = '\n'.join(line for line in text.splitlines() if 'consistency' not in line)
                (directory / filename).write_text(text)
            self.assertEqual(load_config(directory, environ={}), baseline)

    def test_future_observations_share_symmetry_and_skip_missing_terminal(self):
        c = self.config(UNROLL_STEPS='10')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path)
            replay = Replay(read_shard(path, 5), c)
            for seed in range(8):
                obs, actions, _, _, _, _, future, valid = replay.sample(np.random.default_rng(seed))
                for row in range(len(obs)):
                    start = int(obs[row, :2].sum())
                    for step in range(c['UNROLL_STEPS']):
                        exists = start + step + 1 < 9
                        self.assertEqual(bool(valid[row, step]), exists)
                        if not exists:
                            self.assertFalse(future[row, step].any())
                            continue
                        previous = obs[row] if step == 0 else future[row, step - 1]
                        target = future[row, step]
                        np.testing.assert_array_equal(target[Plane.OWN], previous[Plane.OPPONENT])
                        expected = previous[Plane.OWN].copy()
                        expected.flat[actions[row, step]] = 1
                        np.testing.assert_array_equal(target[Plane.OPPONENT], expected)
                        np.testing.assert_array_equal(target[Plane.ON_BOARD], previous[Plane.ON_BOARD])
            disabled = Replay(read_shard(path, 5), dict(c, USE_CONSISTENCY_LOSS=False))
            old = disabled.sample(np.random.default_rng(2))
            new = replay.sample(np.random.default_rng(2))
            self.assertEqual(len(old), 6)
            for a, b in zip(old, new):
                np.testing.assert_array_equal(a, b)

    def test_loss_stop_gradient_and_padding(self):
        head = create_model(self.config()).consistency
        channels = self.config()['HIDDEN_STATE_NUM_CHANNELS']
        predicted = torch.rand(2, channels + 1, 5, 5)
        target = torch.rand_like(predicted)
        predicted[:, -1] = target[:, -1] = 1
        predicted.requires_grad_()
        target.requires_grad_()
        rows = head(predicted, target)
        with torch.no_grad():
            expected = 1 - F.cosine_similarity(head.predictor(head.project(predicted)).flatten(1),
                                                head.project(target).flatten(1), dim=1)
        torch.testing.assert_close(rows, expected)
        rows.mean().backward()
        self.assertGreater(predicted.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        padded_prediction = F.pad(predicted.detach(), (0, 2, 0, 2))
        padded_target = F.pad(target.detach(), (0, 2, 0, 2))
        padded_prediction[:, :-1, 5:] = 100
        padded_target[:, :-1, 5:] = -100
        torch.testing.assert_close(head(padded_prediction, padded_target), rows.detach())

    def test_legacy_optimizer_ema_resume_and_toggle(self):
        enabled = self.config()
        disabled = dict(enabled, USE_CONSISTENCY_LOSS=False)
        device = os.environ.get('MUZERO_TEST_DEVICE', 'cpu')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            game = directory / 'game.mzs'
            write_game(game, weights=[1] * 9)
            records = read_shard(game, 5)
            old = create_model(disabled).to(device)
            old_optimizer = create_optimizer(old, disabled)
            old_average = WeightAverage(old, disabled)
            train_iteration(old, old_optimizer, Replay(records, disabled), disabled, 0, device, old_average)
            checkpoint = directory / 'checkpoint.pt'
            save_checkpoint(checkpoint, old, old_optimizer, disabled, 0, {}, 17, old_average)
            initialize(directory / 'imported', dict(enabled, INIT_MODEL=str(checkpoint)))
            imported = torch.load(checkpoint_path(directory / 'imported', -1), weights_only=True)
            self.assertEqual(imported['training_steps'], 17)
            self.assertEqual(imported['average']['samples'], old_average.samples)
            for key, state in old_optimizer.state_dict()['state'].items():
                for field, value in state.items():
                    torch.testing.assert_close(imported['optimizer']['state'][key][field], value.cpu())
            current = create_model(enabled).to(device)
            optimizer = create_optimizer(current, enabled)
            average = WeightAverage(current, enabled)
            restored = load_checkpoint(checkpoint, current, optimizer, enabled, average)
            self.assertEqual(restored['training_steps'], 17)
            self.assertEqual(average.samples, old_average.samples)
            for name, parameter in old.named_parameters():
                actual = dict(current.named_parameters())[name]
                torch.testing.assert_close(actual, parameter)
                torch.testing.assert_close(average.model.state_dict()[name], old_average.model.state_dict()[name])
                for key, value in old_optimizer.state[parameter].items():
                    torch.testing.assert_close(optimizer.state[actual][key], value)
            for name, value in current.consistency.state_dict().items():
                torch.testing.assert_close(average.model.consistency.state_dict()[name], value)
            before = copy.deepcopy(current.consistency.state_dict())
            result = train_iteration(current, optimizer, Replay(records, enabled), enabled, 1, device, average)
            self.assertGreater(result['consistency_loss'], 0)
            np.testing.assert_allclose(result['loss'], result['policy_loss'] + enabled['VALUE_LOSS_SCALE'] *
                                       result['value_loss'] + enabled['CONSISTENCY_LOSS_SCALE'] * result['consistency_loss'], rtol=1e-6)
            np.testing.assert_allclose(sum(result['step_losses']), result['loss'], rtol=1e-6)
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in current.consistency.state_dict().items()))
            save_checkpoint(checkpoint, current, optimizer, enabled, 1, result, 18, average)
            for config in (enabled, disabled):
                loaded = create_model(config).to(device)
                loaded_optimizer = create_optimizer(loaded, config)
                loaded_average = WeightAverage(loaded, config)
                load_checkpoint(checkpoint, loaded, loaded_optimizer, config, loaded_average)
                for name, parameter in loaded.named_parameters():
                    torch.testing.assert_close(parameter, dict(current.named_parameters())[name])
                    for key, value in optimizer.state[dict(current.named_parameters())[name]].items():
                        torch.testing.assert_close(loaded_optimizer.state[parameter][key], value)
                train_iteration(loaded, loaded_optimizer, Replay(records, config), config, 2, device, loaded_average)
            exported = directory / 'model.pt'
            export_model(current, exported)
            scripted = torch.jit.load(str(exported))
            self.assertFalse(any('consistency' in name for name in scripted.state_dict()))

    def test_checkpoint_rejects_missing_backbone_or_partial_auxiliary(self):
        c = self.config()
        model = create_model(c)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            save_checkpoint(path, model, create_optimizer(model, c), c, 0, {}, 0, WeightAverage(model, c))
            original = torch.load(path, weights_only=True)
            for name in ('representation.start_layer.weight', 'consistency.projector.0.weight'):
                damaged = copy.deepcopy(original)
                del damaged['model'][name]
                torch.save(damaged, path)
                with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                    load_checkpoint(path, model, create_optimizer(model, c), c, WeightAverage(model, c))

    def test_absent_future_observations_do_not_train_head(self):
        c = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path, weights=[0] * 8 + [1])
            model = create_model(c)
            result = train_iteration(model, create_optimizer(model, c), Replay(read_shard(path, 5), c),
                                     c, 0, 'cpu', WeightAverage(model, c))
            self.assertEqual(result['consistency_loss'], 0)
            self.assertTrue(all(p.grad is None for p in model.consistency.parameters()))


@unittest.skipUnless(os.environ.get('MUZERO_TEST_BINARY'), 'Set MUZERO_TEST_BINARY')
class NativeConsistencyTests(unittest.TestCase):
    def test_toggle_in_existing_run_preserves_training_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env = {key: value for key, value in os.environ.items() if key not in KEYS}
            env.update(MUZERO_BINARY=os.environ['MUZERO_TEST_BINARY'], DATA_DIR=str(directory),
                       CONFIG_DIR=str(ROOT / 'configs/minimal_test'), BOARD_SIZES='5',
                       DEVICE=os.environ.get('MUZERO_TEST_DEVICE', 'cpu'))
            for limit, enabled in ((1, False), (3, True), (4, False)):
                env['USE_CONSISTENCY_LOSS'] = str(enabled).lower()
                subprocess.run([sys.executable, '-m', 'muzero.run', str(limit)], env=env,
                               check=True, timeout=180, capture_output=True, text=True)
                checkpoint = torch.load(directory / 'checkpoints/latest.pt', weights_only=True)
                self.assertEqual(checkpoint['training_steps'], limit * 2)
                self.assertEqual(checkpoint['average']['samples'], limit * 8)
                self.assertEqual(any(name.startswith('consistency.') for name in checkpoint['model']), enabled)
                self.assertEqual(len(checkpoint['optimizer']['param_groups']), 2 if enabled else 1)
                for index in checkpoint['optimizer']['param_groups'][0]['params']:
                    self.assertEqual(checkpoint['optimizer']['state'][index]['step'].item(), limit * 2)
                metric = json.loads((directory / 'logs/iters' / f'{limit - 1:08d}.json').read_text())
                self.assertEqual(metric['use_consistency_loss'], enabled)
                self.assertEqual(metric['consistency_loss'] > 0, enabled)
            metric = json.loads((directory / 'logs/iters/00000002.json').read_text())
            self.assertEqual(metric['selfplay_evaluator'], 'network')
            self.assertGreater(metric['games'], 0)
