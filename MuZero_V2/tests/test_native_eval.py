import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from muzero.config import load_config, load_eval_config, render_native_config
from muzero.train import create_model, export_model


@unittest.skipUnless(os.environ.get('MUZERO_TEST_EVAL_BINARY'), 'Set MUZERO_TEST_EVAL_BINARY')
class NativeEvalTests(unittest.TestCase):
    def test_real_model_search_and_illegal_history(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = load_config('configs/minimal_test', environ={'BOARD_SIZES': '7'})
        evaluation = load_eval_config('configs/minimal_test', environ={
            'EVAL_FULL_SEARCH_VISITS': '32',
            'EVAL_DEVICE': os.environ.get('MUZERO_TEST_DEVICE', 'cpu'),
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model, resolved = path / 'model.pt', path / 'eval.cfg'
            export_model(create_model(config), model)
            resolved.write_text(render_native_config(evaluation))
            command = [os.environ['MUZERO_TEST_EVAL_BINARY'], str(resolved), str(model), '5', 'standard']
            output = subprocess.run(command + ['0', '12'], check=True, capture_output=True, text=True, timeout=120)
            result = json.loads(output.stdout)
            self.assertEqual(result['completed_visits'], 32)
            self.assertEqual(sum(c['visits'] for c in result['candidates']), 32)
            self.assertEqual(result['canvas_size'], 7)
            self.assertEqual(result['requests'], 33)
            self.assertAlmostEqual(sum(c['selection_weight'] for c in result['candidates']), 1)
            self.assertAlmostEqual(sum(c['network_prior'] for c in result['candidates']), 1)
            self.assertEqual(result['board'], [1 if i == 0 else -1 if i == 12 else 0 for i in range(25)])
            self.assertEqual(result['action'], result['candidates'][0]['action'])
            for candidate in result['candidates']:
                self.assertTrue(0 <= candidate['action'] < 25)
                self.assertNotIn(candidate['action'], (0, 12))
            for history in (['0', '0'], ['25'], ['-1'], ['2oops'],
                            ['0', '5', '1', '6', '2', '7', '3', '8', '4']):
                bad = subprocess.run(command + history, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
