import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import json
import shutil

import numpy as np
import torch

from muzero.config import ROOT, load_config, render_native_config
from muzero.replay import Replay, read_shard, directory_records
from muzero.train import WeightAverage, create_model, create_optimizer, save_checkpoint, load_checkpoint, export_model
from muzero.prefetch import BatchStream
from muzero.run import ensure_export
from muzero.storage import checkpoint_path
from muzero.network import InferenceModule
from muzero.protocol import INPUT_PLANES, Plane
from test_learning import write_game


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5'})
        torch.set_num_threads(1)

    def test_prefetch_preserves_sampling_and_propagates_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'shard.mzs'
            write_game(path)
            replay = Replay(read_shard(path, 5), self.config)
            rng = np.random.default_rng(23)
            expected = [replay.sample(rng) for _ in range(7)]
            with BatchStream(replay, 23, 7, 2, 'cpu') as batches:
                for arrays in expected:
                    actual = next(batches)
                    for a, b in zip(actual, arrays):
                        np.testing.assert_array_equal(a.numpy(), b)
                with self.assertRaises(StopIteration):
                    next(batches)
            class BrokenReplay:
                def sample(self, rng):
                    raise ValueError('invalid trajectory')
            with BatchStream(BrokenReplay(), 1, 3, 1, 'cpu') as batches:
                with self.assertRaisesRegex(ValueError, 'invalid trajectory'):
                    next(batches)
            with BatchStream(replay, 1, 100, 1, 'cpu') as batches:
                next(batches)
            self.assertFalse(batches.worker.is_alive())

    def test_average_checkpoint_continues_across_iterations(self):
        c = dict(self.config, EMA_HALFLIFE_SAMPLES=8)
        model = create_model(c)
        average = WeightAverage(model, c)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(2)
        average.update(model, 4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(4)
        average.update(model, 8)
        for parameter in average.model.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            save_checkpoint(path, model, create_optimizer(model, c), c, 0, {}, 2, average)
            restored = create_model(c)
            restored_average = WeightAverage(restored, c)
            load_checkpoint(path, restored, create_optimizer(restored, c), c, restored_average)
            self.assertEqual(restored_average.samples, 12)
            restored_average.update(restored, 8)
            for parameter in restored_average.model.parameters():
                torch.testing.assert_close(parameter, torch.full_like(parameter, 3.5))

    def test_export_uses_checkpoint_average_and_replay_snapshot_is_fixed(self):
        c = self.config
        model = create_model(c)
        average = WeightAverage(model, c)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            save_checkpoint(checkpoint_path(data, 0), model, create_optimizer(model, c), c, 0, {}, 1, average)
            exported = torch.jit.load(str(ensure_export(data, c, 0)))
            observation = torch.zeros(1, INPUT_PLANES, 5, 5)
            observation[:, Plane.ON_BOARD] = 1
            with torch.inference_mode():
                for actual, expected in zip(exported.initial(observation), InferenceModule(average.model).initial(observation)):
                    torch.testing.assert_close(actual, expected)
            path = data / 'shard_00000000.mzs'
            write_game(path)
            records = read_shard(path, 5)
            snapshot = data / 'replay/00000000.json'
            Replay(records, c, snapshot)
            original = snapshot.read_bytes()
            Replay(records, c, snapshot)
            self.assertEqual(snapshot.read_bytes(), original)
            state = json.loads(original)
            state['rows'] += 1
            snapshot.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                Replay(records, c, snapshot)
            shutil.copyfile(path, data / 'shard_00000001.mzs')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                directory_records(data, 5)
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                records[0].load()

    def test_failed_export_does_not_replace_published_model(self):
        model = create_model(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            export_model(model, path)
            original = path.read_bytes()
            with torch.no_grad():
                next(model.parameters()).fill_(float('nan'))
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                export_model(model, path)
            self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.environ.get('MUZERO_TEST_BINARY'), 'Requires native selfplay')
    def test_writer_failure_exits_without_blocking_producers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'shard_00000000.mzs.tmp').mkdir()
            config = dict(self.config, SELFPLAY_ROWS_PER_SHARD=1, SELFPLAY_WRITE_QUEUE=1)
            cfg = directory / 'resolved.cfg'
            cfg.write_text(render_native_config(config))
            result = subprocess.run([os.environ['MUZERO_TEST_BINARY'], str(cfg), '-', str(directory),
                                     '20', '42', '0', 'random'], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('selfplay:', result.stderr)
            self.assertEqual(directory_records(directory, 5), [])

    @unittest.skipUnless(os.environ.get('MUZERO_TEST_BINARY'), 'Requires native selfplay')
    def test_shards_preserve_games_and_resume_without_duplicates(self):
        binary = Path(os.environ['MUZERO_TEST_BINARY']).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = directory / 'resolved.cfg'
            c = dict(self.config, SELFPLAY_ROWS_PER_SHARD=40, SELFPLAY_WRITE_QUEUE=1)
            cfg.write_text(render_native_config(c))
            output = directory / 'selfplay'
            def generate(games):
                subprocess.run([str(binary), str(cfg), '-', str(output), str(games), '42', '0', 'random'], check=True, timeout=60)
            generate(4)
            records = directory_records(output, 5)
            self.assertEqual([r.game_id for r in records], list(range(4)))
            self.assertLess(len(list(output.glob('*.mzs'))), 4)
            before = {p: p.read_bytes() for p in output.glob('*.mzs')}
            rows = {r.game_id: r.load().tobytes() for r in records}
            (output / 'shard_99999999.mzs.tmp').write_bytes(b'interrupted')
            generate(7)
            records = directory_records(output, 5)
            self.assertEqual([r.game_id for r in records], list(range(7)))
            for p, data in before.items():
                self.assertEqual(p.read_bytes(), data)
            for r in records:
                if r.game_id in rows:
                    self.assertEqual(r.load().tobytes(), rows[r.game_id])
            path = records[-1].path
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                read_shard(path, 5)


if __name__ == '__main__':
    unittest.main()
