from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from muzero.storage import IterationStore, RunState, checkpoint_path, write_json


class IterationStoreTests(unittest.TestCase):
    def test_recovery_after_commit_repairs_mirrors_and_keeps_committed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            store = IterationStore(data)
            state = RunState(iteration=1, elapsed_seconds=20, target_rows=8, completed_games=2, rows_per_game=4)
            checkpoint_path(data, 0).parent.mkdir()
            checkpoint_path(data, 0).write_bytes(b'previous')
            store.commit(state)
            checkpoint_path(data, 1).write_bytes(b'next')
            model = data / 'models/model_00000002.pt'
            model.parent.mkdir()
            model.write_bytes(b'model')
            record = data / 'selfplay/iter_00000001/shard_00000000.mzs'
            record.parent.mkdir(parents=True)
            record.write_bytes(b'game')
            next_state = dict(state, iteration=2, elapsed_seconds=50, completed_games=4)
            with patch.object(store, 'publish', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    store.commit(next_state)
            self.assertEqual(store.read(), next_state)
            self.assertEqual((data / 'checkpoints/latest.pt').read_bytes(), b'previous')
            store.recover(store.read())
            self.assertEqual((data / 'checkpoints/latest.pt').read_bytes(), b'next')
            self.assertEqual((data / 'models/latest.pt').read_bytes(), b'model')
            self.assertEqual(record.read_bytes(), b'game')
            self.assertFalse(checkpoint_path(data, 0).exists())
            store.recover(store.read())
            self.assertEqual(store.read(), next_state)

    def test_cleanup_can_be_interrupted_and_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            store = IterationStore(data)
            state = RunState(iteration=1, elapsed_seconds=20, target_rows=8, completed_games=2, rows_per_game=4)
            checkpoint_path(data, 0).parent.mkdir()
            checkpoint_path(data, 0).write_bytes(b'committed')
            store.commit(state)
            paths = [data / name for name in (
                'selfplay/iter_00000001/shard_00000000.mzs',
                'selfplay/iter_00000001/shard_00000001.mzs.tmp',
                'replay/00000001.json', 'logs/iters/00000001.json',
                'models/model_00000002.pt', 'checkpoints/checkpoint_00000002.pt',
                'checkpoints/checkpoint_00000002.pt.abc.tmp',
            )]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'uncommitted')
            original = Path.unlink

            def interrupt(path, *args, **kwargs):
                original(path, *args, **kwargs)
                if path == data / 'replay/00000001.json':
                    raise KeyboardInterrupt

            with patch.object(Path, 'unlink', interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    store.recover(state)
            self.assertEqual(store.read(), state)
            store.recover(state)
            self.assertTrue(all(not path.exists() for path in paths))
            self.assertEqual((data / 'checkpoints/latest.pt').read_bytes(), b'committed')
            self.assertEqual(store.read(), state)

    def test_missing_committed_checkpoint_does_not_delete_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            store = IterationStore(data)
            state = RunState(iteration=1, elapsed_seconds=20, target_rows=8, completed_games=2, rows_per_game=4)
            write_json(store.state_path, state)
            path = checkpoint_path(data, 1)
            path.parent.mkdir()
            path.write_bytes(b'uncommitted')
            with self.assertRaisesRegex(ValueError, 'Missing committed checkpoint'):
                store.recover(state)
            self.assertEqual(path.read_bytes(), b'uncommitted')


if __name__ == '__main__':
    unittest.main()
