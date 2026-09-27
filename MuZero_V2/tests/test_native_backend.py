import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from muzero.config import ROOT, load_config
from muzero.train import create_model, export_model


@unittest.skipUnless(os.environ.get('MUZERO_TEST_BACKEND_BINARY'), 'Set MUZERO_TEST_BACKEND_BINARY')
class NativeBackendTests(unittest.TestCase):
    def test_reference_outputs_and_retained_latents(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        config = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            export_model(create_model(config), path)
            subprocess.run([os.environ['MUZERO_TEST_BACKEND_BINARY'], str(path),
                            os.environ.get('MUZERO_TEST_DEVICE', 'cpu'), '5'], check=True, timeout=120)


if __name__ == '__main__':
    unittest.main()
