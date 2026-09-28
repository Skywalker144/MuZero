from pathlib import Path
import tempfile
import unittest

import torch

from muzero.config import ROOT, load_config
from muzero.experiment import prepare_initializations
from muzero.replay import read_shard
from muzero.run import initialize, train_and_checkpoint
from muzero.storage import checkpoint_path
from test_learning import write_game


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'configs/minimal_test', environ={
            'MIN_ROWS': '18', 'BOARD_SIZES': '5', 'BOARD_SIZE_WEIGHTS': '1',
            'RULES': 'freestyle', 'RULE_WEIGHTS': '1',
        })
        torch.set_num_threads(1)

    def test_training_steps_track_updates_not_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            c = self.config
            initialize(data, c)
            initial = torch.load(checkpoint_path(data, -1), weights_only=True)
            self.assertEqual(initial['training_steps'], 0)
            train_and_checkpoint(data, c, 0, [])
            cold = torch.load(checkpoint_path(data, 0), weights_only=True)
            self.assertEqual(cold['iteration'], 0)
            self.assertEqual(cold['training_steps'], 0)
            for key in initial['model']:
                self.assertTrue(torch.equal(initial['model'][key], cold['model'][key]))
            records = []
            for index in range(2):
                path = data / f'game_{index}.mzs'
                write_game(path)
                records.append(read_shard(path, 5)[0])
            train_and_checkpoint(data, c, 1, records)
            trained = torch.load(checkpoint_path(data, 1), weights_only=True)
            self.assertEqual(trained['training_steps'], c['TRAIN_STEPS'])
            self.assertTrue(any(not torch.equal(cold['model'][key], trained['model'][key]) for key in cold['model']))
            train_and_checkpoint(data, c, 2, [])
            self.assertEqual(torch.load(checkpoint_path(data, 2), weights_only=True)['training_steps'], c['TRAIN_STEPS'])
            imported = data / 'imported'
            initialize(imported, dict(c, INIT_MODEL=str(checkpoint_path(data, 2))))
            self.assertEqual(torch.load(checkpoint_path(imported, -1), weights_only=True)['training_steps'], c['TRAIN_STEPS'])

    def test_shared_initialization_is_untrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            c = dict(self.config)
            prepare_initializations(data, [{'config': c}])
            self.assertTrue(c['INIT_MODEL'])
            initialize(data / 'run', c)
            checkpoint = torch.load(checkpoint_path(data / 'run', -1), weights_only=True)
            self.assertEqual(checkpoint['training_steps'], 0)

    def test_unknown_initialization_training_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            c = dict(self.config)
            prepare_initializations(data, [{'config': c}])
            initial = torch.load(c['INIT_MODEL'], weights_only=True)
            initial.pop('training_steps', None)
            torch.save(initial, c['INIT_MODEL'])
            with self.assertRaisesRegex(ValueError, 'training_steps'):
                initialize(data / 'run', c)


if __name__ == '__main__':
    unittest.main()
