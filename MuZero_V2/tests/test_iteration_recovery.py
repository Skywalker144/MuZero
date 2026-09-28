import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from muzero.config import KEYS, ROOT, load_config
from muzero.process import run_process
from muzero.replay import catalog
from muzero.run import run, train_and_checkpoint, ensure_export
from muzero.storage import write_json


@unittest.skipUnless(os.environ.get('MUZERO_TEST_BINARY'), 'Set MUZERO_TEST_BINARY')
class IterationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.binary = Path(os.environ['MUZERO_TEST_BINARY']).resolve()
        self.config = load_config(ROOT / 'configs/minimal_test', environ={
            'DEVICE': os.environ.get('MUZERO_TEST_DEVICE', 'cpu'),
            'BOARD_SIZES': '5', 'BOARD_SIZE_WEIGHTS': '1',
            'RULES': 'freestyle', 'RULE_WEIGHTS': '1',
            'SELFPLAY_SCHEDULE': 'fixed', 'GAMES_PER_ITER': '2',
        })

    def test_interruptions_discard_whole_iteration_and_time(self):
        for phase in ('selfplay', 'optimizer', 'checkpoint', 'export', 'commit'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                data = Path(tmp)
                config = dict(self.config, DATA_DIR=str(data))
                clock = SimpleNamespace(now=0.0)
                clock.monotonic = lambda: clock.now

                def selfplay(command):
                    run_process(command)
                    clock.now += 10
                    if interrupt[0] == 'selfplay':
                        raise KeyboardInterrupt

                def train(*args):
                    result = train_and_checkpoint(*args)
                    clock.now += 20
                    if interrupt[0] == 'checkpoint':
                        raise KeyboardInterrupt
                    return result

                def export(*args):
                    result = ensure_export(*args)
                    if interrupt[0] == 'export' and args[2] == 1:
                        raise KeyboardInterrupt
                    return result

                def save(path, value):
                    if interrupt[0] == 'commit' and path == data / 'logs/state.json' and value['iteration'] == 2:
                        raise KeyboardInterrupt
                    write_json(path, value)

                step = torch.optim.AdamW.step

                def update(optimizer, *args, **kwargs):
                    result = step(optimizer, *args, **kwargs)
                    if interrupt[0] == 'optimizer':
                        clock.now += 15
                        raise KeyboardInterrupt
                    return result

                interrupt = [None]
                with patch('muzero.run.time', clock), patch('muzero.run.run_process', side_effect=selfplay), \
                        patch('muzero.run.train_and_checkpoint', side_effect=train), \
                        patch('muzero.run.ensure_export', side_effect=export), \
                        patch('torch.optim.AdamW.step', update), patch('muzero.storage.write_json', side_effect=save), \
                        patch('muzero.run.write_json', side_effect=save):
                    run(config, self.binary, 1)
                    state_path = data / 'logs/state.json'
                    committed = state_path.read_bytes()
                    checkpoint = (data / 'checkpoints/latest.pt').read_bytes()
                    model = (data / 'models/latest.pt').read_bytes()
                    history = {p: p.read_bytes() for p in (data / 'selfplay/iter_00000000').iterdir()}
                    self.assertEqual(json.loads(committed)['elapsed_seconds'], 30)
                    interrupt[0] = phase
                    with self.assertRaises(KeyboardInterrupt):
                        run(config, self.binary, 2)
                    self.assertEqual(state_path.read_bytes(), committed)
                    interrupt[0] = None
                    clock.now += 86400
                    run(config, self.binary, 1)
                    self.assertEqual(state_path.read_bytes(), committed)
                    self.assertEqual((data / 'checkpoints/latest.pt').read_bytes(), checkpoint)
                    self.assertEqual((data / 'models/latest.pt').read_bytes(), model)
                    self.assertFalse((data / 'selfplay/iter_00000001').exists())
                    self.assertFalse((data / 'replay/00000001.json').exists())
                    self.assertFalse((data / 'logs/iters/00000001.json').exists())
                    self.assertFalse((data / 'models/model_00000002.pt').exists())
                    self.assertEqual({p: p.read_bytes() for p in history}, history)
                    run(config, self.binary, 2)
                    state = json.loads(state_path.read_text())
                    metric = json.loads((data / 'logs/iters/00000001.json').read_text())
                    self.assertEqual(state['elapsed_seconds'], 60)
                    self.assertEqual(metric['elapsed_seconds'], 60)
                    self.assertEqual(len(catalog(data, 5)), 4)
                    trained = torch.load(data / 'checkpoints/latest.pt', weights_only=True)
                    self.assertEqual(trained['training_steps'], 2 * config['TRAIN_STEPS'])
                    run(dict(config, MAX_TIME_SECONDS=60), self.binary, 100)
                    self.assertEqual(json.loads(state_path.read_text()), state)
                    self.assertFalse((data / 'selfplay/iter_00000002').exists())

    def test_old_commit_format_is_rejected_without_changing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            config = dict(self.config, DATA_DIR=str(data))
            run(config, self.binary, 0)
            path = data / 'logs/manifest.json'
            manifest = json.loads(path.read_text())
            del manifest['iteration_commit_version']
            write_json(path, manifest)
            original = {p: p.read_bytes() for p in data.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError, 'commit format; use a new DATA_DIR'):
                run(config, self.binary, 1)
            self.assertEqual({p: p.read_bytes() for p in original}, original)

    def test_sigint_runner_and_experiment_preserve_committed_budget(self):
        for experiment in (False, True):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / 'data/arm'
                umbrella = root / 'configs'
                arm = umbrella / 'arm'
                arm.mkdir(parents=True)
                (umbrella / 'exp.cfg').write_text(
                    '[experiment]\nmax_iters = 1\nmax_time_seconds = 0\nshared_init = false\narm_gpus =\n')
                (arm / 'run.cfg').write_text(
                    f'extends = {ROOT / "configs/minimal_test"}\n[run]\ndata_dir = {data}\n')
                env = {key: value for key, value in os.environ.items() if key not in KEYS}
                env.update(MUZERO_BINARY=str(self.binary), DEVICE=self.config['DEVICE'], BOARD_SIZES='5',
                           GAMES_PER_ITER='100000', SELFPLAY_SCHEDULE='fixed')
                if experiment:
                    env['CONFIG_DIR'] = str(umbrella)
                    command = [sys.executable, '-c',
                               'from pathlib import Path; import sys; import muzero.experiment as e; '
                               'e.ROOT = Path(sys.argv.pop()); e.main()', str(root)]
                else:
                    env['CONFIG_DIR'] = str(arm)
                    command = [sys.executable, '-m', 'muzero.run', '1']
                output_path = root / 'output.log'
                with output_path.open('w') as output:
                    process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
                    try:
                        deadline = time.monotonic() + 60
                        while not list(data.glob('selfplay/iter_00000000/shard_*.mzs')):
                            if process.poll() is not None:
                                self.fail(output_path.read_text())
                            if time.monotonic() >= deadline:
                                self.fail('Selfplay did not publish a shard before the interrupt deadline')
                            time.sleep(0.05)
                        state_path = data / 'logs/state.json'
                        before = state_path.read_bytes()
                        process.send_signal(signal.SIGINT)
                        self.assertNotEqual(process.wait(timeout=40), 0)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                self.assertEqual(state_path.read_bytes(), before)
                self.assertEqual(json.loads(before)['elapsed_seconds'], 0)
                env['CONFIG_DIR'] = str(arm)
                result = subprocess.run([sys.executable, '-m', 'muzero.run', '0'], env=env,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse((data / 'selfplay/iter_00000000').exists())
                self.assertEqual(state_path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
