import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from muzero.config import KEYS, ROOT
from muzero.replay import directory_records


@unittest.skipUnless(os.environ.get('MUZERO_TEST_BINARY'), 'Set MUZERO_TEST_BINARY to the built LibTorch executable')
class NativePipelineTests(unittest.TestCase):
    def test_selfplay_training_export_and_commit_recovery(self):
        binary = Path(os.environ['MUZERO_TEST_BINARY']).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            env = {key: value for key, value in os.environ.items() if key not in KEYS}
            env.update({'MUZERO_BINARY': str(binary), 'DATA_DIR': str(data),
                        'CONFIG_DIR': str(ROOT / 'configs/minimal_test'), 'DEVICE': os.environ.get('MUZERO_TEST_DEVICE', 'cpu'), 'BOARD_SIZES': '5'})
            command = [sys.executable, '-m', 'muzero.run', '1']
            subprocess.run(command, env=env, check=True, timeout=180)
            state_path = data / 'logs/state.json'
            state = json.loads(state_path.read_text())
            self.assertEqual(state['iteration'], 1)
            metric = json.loads((data / 'logs/iters/00000000.json').read_text())
            self.assertEqual(metric['elapsed_seconds'], state['elapsed_seconds'])
            plot = data / 'training.png'
            self.assertEqual(plot.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            records = directory_records(data / 'selfplay/iter_00000000', 5)
            self.assertEqual(len(records), 2)
            checkpoint = data / 'checkpoints/latest.pt'
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint.unlink()
            (data / 'models/latest.pt').unlink()
            subprocess.run(command, env=env, check=True, timeout=180)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads(state_path.read_text())['iteration'], 1)
            self.assertEqual(len(directory_records(data / 'selfplay/iter_00000000', 5)), 2)
            self.assertTrue((data / 'models/model_00000001.pt').exists())
            plot.unlink()
            subprocess.run(command, env=env, check=True, timeout=180)
            self.assertEqual(plot.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)
            self.assertEqual(len(list((data / 'logs/iters').glob('*.json'))), 1)

    def test_mixed_rules_sizes_native_training(self):
        import numpy as np
        import torch
        from muzero.config import load_config, render_native_config
        from muzero.protocol import Rule
        from muzero.replay import Replay, placement_mask
        from muzero.train import WeightAverage, create_model, create_optimizer, export_model, train_iteration
        from test_network import ARCHITECTURE

        binary = Path(os.environ['MUZERO_TEST_BINARY']).resolve()
        c = load_config(ROOT / 'configs/minimal_test', environ=ARCHITECTURE | {
            'DEVICE': os.environ.get('MUZERO_TEST_DEVICE', 'cpu'),
            'BOARD_SIZES': '5,6', 'BOARD_SIZE_WEIGHTS': '1,1',
            'RULES': 'freestyle,standard,renju', 'RULE_WEIGHTS': '1,1,1',
            'BATCH_SIZE': '4', 'TRAIN_STEPS': '1', 'NUM_GAME_THREADS': '2',
        })
        torch.set_num_threads(c['TORCH_THREADS'])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = create_model(c).to(c['DEVICE'])
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
                    subprocess.run([str(binary), str(resolved), str(deployed), str(games), '2', '42', '0', 'network'],
                                   check=True, timeout=180)
                    for record in directory_records(games, c['CANVAS_SIZE']):
                        self.assertEqual((record.size, record.rule), (size, rule))
                        data = record.load()
                        self.assertTrue((data['policy'][~placement_mask(data['observation']).reshape(record.rows, -1)] == 0).all())
                        records.append(record)
            resolved.write_text(render_native_config(c))
            mixed = directory / 'mixed'
            subprocess.run([str(binary), str(resolved), str(deployed), str(mixed), '12', '71', '0', 'network'],
                           check=True, timeout=180)
            for record in directory_records(mixed, c['CANVAS_SIZE']):
                self.assertIn(record.size, c['BOARD_SIZES'])
                self.assertIn(record.rule.name.lower(), c['RULES'])
                record.load()
                records.append(record)
            replay = Replay(records, c)
            optimizer = create_optimizer(model, c)
            metrics = train_iteration(model, optimizer, replay, c, 0, c['DEVICE'], WeightAverage(model, c))
            self.assertTrue(np.isfinite(metrics['loss']))
            export_model(model, deployed)
            subprocess.run([str(binary), str(resolved), str(deployed), str(directory / 'trained'), '1', '99', '0', 'network'],
                           check=True, timeout=180)


if __name__ == '__main__':
    unittest.main()
