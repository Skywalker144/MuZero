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
                        'CONFIG_DIR': str(ROOT / 'configs/minimal_test'), 'DEVICE': 'cpu'})
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


if __name__ == '__main__':
    unittest.main()
