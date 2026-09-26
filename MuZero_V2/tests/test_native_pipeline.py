import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from muzero.config import KEYS, ROOT


@unittest.skipUnless(os.environ.get('MUZERO_TEST_BINARY'), 'Set MUZERO_TEST_BINARY to the built LibTorch executable')
class NativePipelineTests(unittest.TestCase):
    def test_selfplay_training_export_and_commit_recovery(self):
        binary = Path(os.environ['MUZERO_TEST_BINARY']).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            env = {key: value for key, value in os.environ.items() if key not in KEYS}
            env.update({'MUZERO_BINARY': str(binary), 'DATA_DIR': str(data),
                        'CONFIG_DIR': str(ROOT / 'configs/minimal_test'), 'DEVICE': 'cpu', 'BOARD_SIZES': '5'})
            command = [sys.executable, '-m', 'muzero.run', '1']
            subprocess.run(command, env=env, check=True, timeout=180)
            state_path = data / 'logs/state.json'
            state = json.loads(state_path.read_text())
            self.assertEqual(state['iteration'], 1)
            self.assertIsNone(state['pending'])
            records = list((data / 'selfplay/iter_00000000').glob('*.mzg'))
            self.assertEqual(len(records), 2)
            checkpoint = data / 'checkpoints/latest.pt'
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state.update(iteration=0, completed_games=0, pending={
                'games': 2, 'seed': 0, 'selfplay_seconds': 0,
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            })
            state_path.write_text(json.dumps(state))
            subprocess.run(command, env=env, check=True, timeout=180)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads(state_path.read_text())['iteration'], 1)
            self.assertEqual(len(list((data / 'selfplay/iter_00000000').glob('*.mzg'))), 2)
            self.assertTrue((data / 'models/model_00000001.pt').exists())

    def test_mixed_rules_sizes_native_training(self):
        import numpy as np
        import torch
        from muzero.config import load_config, render_native_config
        from muzero.protocol import Rule
        from muzero.replay import GameRecord, Replay, placement_mask
        from muzero.train import create_model, create_optimizer, export_model, train_iteration

        binary = Path(os.environ['MUZERO_TEST_BINARY']).resolve()
        c = load_config(ROOT / 'configs/minimal_test', environ={
            'BOARD_SIZES': '5,6', 'BOARD_SIZE_WEIGHTS': '1,1',
            'RULES': 'freestyle,standard,renju', 'RULE_WEIGHTS': '1,1,1',
            'BATCH_SIZE': '4', 'TRAIN_STEPS': '1', 'NUM_GAME_THREADS': '2',
        })
        torch.set_num_threads(c['TORCH_THREADS'])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = create_model(c)
            deployed = directory / 'model.pt'
            export_model(model, deployed)
            records = []
            for size in c['BOARD_SIZES']:
                for rule in Rule:
                    resolved = directory / 'resolved.cfg'
                    selected = dict(c, BOARD_SIZES=[size, 6] if size != 6 else [6],
                                    BOARD_SIZE_WEIGHTS=[1, 0] if size != 6 else [1],
                                    RULES=[rule.name.lower()], RULE_WEIGHTS=[1])
                    resolved.write_text(render_native_config(selected))
                    games = directory / f'{size}_{rule.name}'
                    subprocess.run([str(binary), str(resolved), str(deployed), str(games), '2', '42', '0'],
                                   check=True, timeout=180)
                    for path in sorted(games.glob('*.mzg')):
                        record = GameRecord.inspect(path, c['CANVAS_SIZE'])
                        self.assertEqual((record.size, record.rule), (size, rule))
                        data = record.load()
                        self.assertTrue((data['policy'][~placement_mask(data['observation']).reshape(record.rows, -1)] == 0).all())
                        records.append(record)
            resolved.write_text(render_native_config(c))
            mixed = directory / 'mixed'
            subprocess.run([str(binary), str(resolved), str(deployed), str(mixed), '12', '71', '0'],
                           check=True, timeout=180)
            for path in sorted(mixed.glob('*.mzg')):
                record = GameRecord.inspect(path, c['CANVAS_SIZE'])
                self.assertIn(record.size, c['BOARD_SIZES'])
                self.assertIn(record.rule.name.lower(), c['RULES'])
                record.load()
                records.append(record)
            replay = Replay(records, c)
            optimizer = create_optimizer(model, c)
            metrics = train_iteration(model, optimizer, replay, c, 0, 'cpu')
            self.assertTrue(np.isfinite(metrics['loss']))
            export_model(model, deployed)
            subprocess.run([str(binary), str(resolved), str(deployed), str(directory / 'trained'), '1', '99', '0'],
                           check=True, timeout=180)


if __name__ == '__main__':
    unittest.main()
