import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from muzero.config import ROOT, load_config, render_native_config
from muzero.process import run_process
from muzero.replay import directory_records
from muzero.run import run


@unittest.skipUnless(os.environ.get('MUZERO_TEST_BINARY'), 'Set MUZERO_TEST_BINARY to the built LibTorch executable')
class NativeBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.binary = Path(os.environ['MUZERO_TEST_BINARY']).resolve()
        self.config = load_config(ROOT / 'configs/minimal_test', environ={
            'DEVICE': os.environ.get('MUZERO_TEST_DEVICE', 'cpu'),
            'BOARD_SIZES': '5', 'BOARD_SIZE_WEIGHTS': '1', 'RULES': 'freestyle', 'RULE_WEIGHTS': '1',
            'SELFPLAY_SCHEDULE': 'fixed', 'GAMES_PER_ITER': '1', 'MIN_ROWS': '26',
        })

    def test_threshold_switch_export_recovery_and_pretrained_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            c = dict(self.config, DATA_DIR=str(data / 'source'))
            source = Path(c['DATA_DIR'])
            run(c, self.binary, 1)
            self.assertFalse((source / 'models').exists())
            checkpoint_path = source / 'checkpoints/latest.pt'
            self.assertEqual(torch.load(checkpoint_path, weights_only=True)['training_steps'], 0)

            def fail_export(*args):
                raise RuntimeError('export interrupted')

            with patch('muzero.run.ensure_export', side_effect=fail_export):
                with self.assertRaisesRegex(RuntimeError, 'export interrupted'):
                    run(c, self.binary, 4)
            committed = checkpoint_path.read_bytes()
            state = json.loads((source / 'logs/state.json').read_text())
            self.assertEqual(state['pending']['evaluator'], 'random')
            self.assertGreater(torch.load(checkpoint_path, weights_only=True)['training_steps'], 0)
            run(c, self.binary, state['iteration'] + 1)
            self.assertEqual(checkpoint_path.read_bytes(), committed)
            self.assertTrue((source / 'models/latest.pt').exists())
            run(c, self.binary, 4)
            metrics = [json.loads(path.read_text()) for path in sorted((source / 'logs/iters').glob('*.json'))]
            self.assertEqual(metrics[0]['selfplay_evaluator'], 'random')
            self.assertEqual(metrics[3]['selfplay_evaluator'], 'network')
            first_update = next(i for i, metric in enumerate(metrics) if metric['steps'])
            self.assertTrue(all(metric['selfplay_evaluator'] == 'random' for metric in metrics[:first_update + 1]))
            self.assertTrue(all(metric['selfplay_evaluator'] == 'network' for metric in metrics[first_update + 1:]))
            trained = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(trained['training_steps'], sum(metric['steps'] for metric in metrics))
            imported = dict(c, DATA_DIR=str(data / 'imported'), INIT_MODEL=str(checkpoint_path))
            run(imported, self.binary, 1)
            metric = json.loads((data / 'imported/logs/iters/00000000.json').read_text())
            self.assertEqual(metric['selfplay_evaluator'], 'network')

    def test_partial_random_iteration_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            c = dict(self.config, DATA_DIR=str(data / 'resumed'), GAMES_PER_ITER=3, MIN_ROWS=1000)

            def interrupt_selfplay(command):
                partial = list(command)
                partial[4] = '1'
                run_process(partial)
                raise RuntimeError('selfplay interrupted')

            with patch('muzero.run.run_process', side_effect=interrupt_selfplay):
                with self.assertRaisesRegex(RuntimeError, 'selfplay interrupted'):
                    run(c, self.binary, 1)
            state = json.loads((data / 'resumed/logs/state.json').read_text())
            self.assertEqual(state['pending']['evaluator'], 'random')
            self.assertEqual(state['pending']['model_generation'], -1)
            existing = next((data / 'resumed/selfplay/iter_00000000').glob('*.mzs'))
            original = existing.read_bytes()
            run(dict(c, NUM_GAME_THREADS=4), self.binary, 1)
            self.assertEqual(existing.read_bytes(), original)
            run(dict(c, DATA_DIR=str(data / 'continuous'), NUM_GAME_THREADS=1), self.binary, 1)
            resumed = directory_records(data / 'resumed/selfplay/iter_00000000', 5)
            continuous = directory_records(data / 'continuous/selfplay/iter_00000000', 5)
            self.assertEqual(len(resumed), 3)
            self.assertEqual([record.load().tobytes() for record in resumed], [record.load().tobytes() for record in continuous])

    def test_adaptive_budget_starts_after_bootstrap_and_survives_partial_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            c = dict(self.config, DATA_DIR=str(data), SELFPLAY_SCHEDULE='adaptive',
                     BOOTSTRAP_GAMES=1, BACKFILL_FACTOR=4.0, MIN_ROWS=60, REPLAY_RATIO=0.08)
            run(c, self.binary, 2)
            metrics_dir = data / 'logs/iters'
            bootstrap = json.loads((metrics_dir / '00000001.json').read_text())
            self.assertGreater(bootstrap['total_rows'], c['MIN_ROWS'])
            self.assertEqual(bootstrap['steps'], c['TRAIN_STEPS'])
            increment = c['TRAIN_STEPS'] * c['BATCH_SIZE'] / c['REPLAY_RATIO']
            average = bootstrap['rows'] / bootstrap['games']
            expected_games = math.ceil(increment / average)
            self.assertGreater(expected_games, 1)

            def interrupt_selfplay(command):
                partial = list(command)
                partial[4] = '1'
                run_process(partial)
                raise RuntimeError('selfplay interrupted')

            with patch('muzero.run.run_process', side_effect=interrupt_selfplay):
                with self.assertRaisesRegex(RuntimeError, 'selfplay interrupted'):
                    run(c, self.binary, 3)
            state_path = data / 'logs/state.json'
            pending = json.loads(state_path.read_text())
            self.assertEqual(pending['iteration'], 2)
            self.assertEqual(pending['target_rows'], bootstrap['total_rows'] + increment)
            self.assertEqual(pending['pending']['games'], expected_games)
            self.assertEqual(pending['pending']['evaluator'], 'network')
            existing = next((data / 'selfplay/iter_00000002').glob('*.mzs'))
            original = existing.read_bytes()
            run(c, self.binary, 4)
            self.assertEqual(existing.read_bytes(), original)
            second = json.loads((metrics_dir / '00000002.json').read_text())
            third = json.loads((metrics_dir / '00000003.json').read_text())
            self.assertEqual(second['games'], expected_games)
            next_target = bootstrap['total_rows'] + 2 * increment
            self.assertEqual(third['games'], max(1, math.ceil(
                (next_target - second['total_rows']) / (second['rows'] / second['games']))))
            self.assertEqual(second['steps'], c['TRAIN_STEPS'])
            self.assertEqual(third['steps'], c['TRAIN_STEPS'])
            self.assertEqual(third['selfplay_evaluator'], 'network')
            state = json.loads(state_path.read_text())
            self.assertEqual(state['target_rows'], next_target)
            self.assertEqual(state['iteration'], 4)
            self.assertIsNone(state['pending'])

    def test_random_native_has_no_model_or_cuda_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            for rule in ('freestyle', 'standard', 'renju'):
                c = dict(self.config, DEVICE='cuda:999', BOARD_SIZES=[5, 6], BOARD_SIZE_WEIGHTS=[1, 1],
                         CANVAS_SIZE=6, RULES=[rule], RULE_WEIGHTS=[1])
                cfg = data / 'resolved.cfg'
                cfg.write_text(render_native_config(c))
                output = data / rule
                result = subprocess.run([str(self.binary), str(cfg), str(data / 'missing.pt'), str(output),
                                         '4', '42', '0', 'random'], check=True, capture_output=True, text=True, timeout=60)
                self.assertIn('evaluator=random nn_requests=0 nn_batches=0', result.stdout)
                for record in directory_records(output, 6):
                    record.load()

    def test_network_override_uses_untrained_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            c = dict(self.config, DATA_DIR=str(data), BOOTSTRAP_EVALUATOR='network')
            run(c, self.binary, 1)
            metric = json.loads((data / 'logs/iters/00000000.json').read_text())
            self.assertEqual(metric['selfplay_evaluator'], 'network')
            self.assertEqual(metric['steps'], 0)
            self.assertTrue((data / 'models/model_00000000.pt').exists())


if __name__ == '__main__':
    unittest.main()
