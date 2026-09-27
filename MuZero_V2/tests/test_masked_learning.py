from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from muzero.config import ROOT, load_config
from muzero.network import InferenceModule
from muzero.protocol import INPUT_PLANES, Plane, Rule
from muzero.replay import read_shard, Replay, augment, placement_mask
from muzero.train import WeightAverage, create_model, create_optimizer, export_model, train_iteration
from test_learning import write_game


class MaskedLearningTests(unittest.TestCase):
    def config(self, sizes='5,9'):
        return load_config(ROOT / 'configs/minimal_test', environ={
            'BOARD_SIZES': sizes, 'BOARD_SIZE_WEIGHTS': ','.join('1' for _ in sizes.split(',')),
            'RULES': 'freestyle,standard,renju', 'RULE_WEIGHTS': '1,1,1',
            'BATCH_SIZE': '4', 'TRAIN_STEPS': '1',
        })

    def test_mixed_replay_optimizer_and_export(self):
        c = self.config()
        torch.set_num_threads(c['TORCH_THREADS'])
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for size in c['BOARD_SIZES']:
                for rule in Rule:
                    path = Path(tmp) / f'{size}_{rule.name}.mzs'
                    write_game(path, size, 9, rule)
                    records.append(read_shard(path, 9)[0])
            replay = Replay(records, c)
            obs, actions, policies, _, _ = replay.sample(np.random.default_rng(0))
            self.assertEqual(obs.shape, (4, INPUT_PLANES, 9, 9))
            on_board = obs[:, Plane.ON_BOARD].reshape(4, 81)
            self.assertTrue((policies * (1 - on_board[:, None, None])).sum() == 0)
            self.assertTrue(on_board[np.arange(4)[:, None], actions].all())
            model = create_model(c)
            metrics = train_iteration(model, create_optimizer(model, c), replay, c, 0, 'cpu', WeightAverage(model, c))
            self.assertTrue(np.isfinite(metrics['loss']))
            export = Path(tmp) / 'model.pt'
            export_model(model, export)
            scripted = torch.jit.load(str(export))
            hidden, logits, _ = scripted.initial(torch.from_numpy(obs))
            for step in range(c['UNROLL_STEPS']):
                hidden, logits, _ = scripted.recurrent(hidden, torch.from_numpy(actions[:, step]))
                torch.testing.assert_close(hidden[:, -1], torch.from_numpy(obs[:, Plane.ON_BOARD]))
                self.assertTrue((logits.softmax(-1).detach().numpy()[on_board == 0] == 0).all())

    def test_padding_invariance_through_dynamics(self):
        torch.set_num_threads(1)
        small = create_model(self.config('5')).eval()
        large = create_model(self.config('9')).eval()
        large.load_state_dict(small.state_dict())
        obs = torch.zeros(2, INPUT_PLANES, 5, 5)
        obs[:, Plane.ON_BOARD] = 1
        obs[:, Plane.BLACK_TO_MOVE] = 1
        obs[0, Plane.RENJU] = 1
        obs[1, Plane.STANDARD] = 1
        obs[:, Plane.OWN, 2, 2] = 1
        padded = torch.zeros(2, INPUT_PLANES, 9, 9)
        padded[:, :, :5, :5] = obs
        a = InferenceModule(small).initial(obs)
        b = InferenceModule(large).initial(padded)
        for step in range(3):
            torch.testing.assert_close(a[0], b[0][:, :, :5, :5], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(a[1].reshape(2, 5, 5), b[1].reshape(2, 9, 9)[:, :5, :5], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(a[2], b[2], atol=2e-5, rtol=2e-5)
            a = InferenceModule(small).recurrent(a[0], torch.tensor([step * 5 + 1] * 2))
            b = InferenceModule(large).recurrent(b[0], torch.tensor([step * 9 + 1] * 2))

    def test_symmetry_and_record_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'game.mzs'
            write_game(path, 5, 9, Rule.STANDARD)
            record = read_shard(path, 9)[0]
            data = record.load()
            obs = data['observation'][:8].astype(np.float32)
            actions = data['action'][:8, None].astype(np.int64)
            policies = data['policy'][:8, None, None]
            transformed, moved, targets = augment(obs, actions, policies, np.random.default_rng(4))
            legal = placement_mask(transformed).reshape(8, 81)
            self.assertTrue(legal[np.arange(8), moved[:, 0]].all())
            self.assertTrue((targets[:, 0, 0][~legal] == 0).all())
            self.assertTrue((transformed[:, Plane.STANDARD] == transformed[:, Plane.ON_BOARD]).all())
            with self.assertRaises(ValueError):
                read_shard(path, 5)[0]
            raw = bytearray(path.read_bytes())
            raw[:8] = b'MZV2GAME'
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                read_shard(path, 9)[0]
