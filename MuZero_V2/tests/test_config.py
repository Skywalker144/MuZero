import tempfile
import unittest
from pathlib import Path

from muzero.config import CONFIG_FILES, ROOT, load_config, read_profile, render_config


class ConfigTests(unittest.TestCase):
    def test_baseline_and_delta(self):
        baseline = load_config(ROOT / 'configs/baseline', environ={})
        experiment = load_config(ROOT / 'configs/exp_baseline', environ={})
        original = load_config(ROOT / 'configs/muzero', environ={})
        self.assertEqual(baseline['BOARD_SIZES'], [15])
        self.assertEqual(experiment['BOARD_SIZES'], [11])
        self.assertEqual(baseline['FULL_SEARCH_VISITS'], 400)
        self.assertEqual(experiment['FULL_SEARCH_VISITS'], 220)
        self.assertEqual(baseline['DIRICHLET_TOTAL_CONCENTRATION'], 6.75)
        self.assertEqual(experiment['DIRICHLET_TOTAL_CONCENTRATION'], 3.63)
        self.assertEqual(experiment['VALUE_LOSS_SCALE'], baseline['VALUE_LOSS_SCALE'])
        self.assertEqual(original['VALUE_HEAD'], 'scalar')
        for key in ('AUXILIARY_POLICY_HEADS', 'USE_FPU', 'USE_LCB_FOR_SELECTION',
                    'SHAPED_DIRICHLET_NOISE', 'SYMMETRY_AUGMENTATION', 'CHEAP_SEARCH_PROB'):
            self.assertFalse(original[key], key)
        self.assertEqual(original['REPLAY_WINDOW'], 'fixed')
        self.assertEqual(original['SELFPLAY_SCHEDULE'], 'fixed')

    def test_local_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'run.cfg').write_text('extends = baseline\n')
            (path / 'selfplay.cfg').write_text('[search]\nfull_search_visits = 40\n')
            (path / 'selfplay.cfg.local').write_text('[search]\nfull_search_visits = 30\n')
            self.assertEqual(load_config(path, environ={})['FULL_SEARCH_VISITS'], 30)
            self.assertEqual(load_config(path, environ={'FULL_SEARCH_VISITS': '10'})['FULL_SEARCH_VISITS'], 10)

    def test_invalid_cycle_unknown_and_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'run.cfg').write_text(f'extends = {path}\n')
            with self.assertRaises(ValueError):
                load_config(path, environ={})
            for text in ('[search]\ntypo = 1\n', '[run]\nfull_search_visits = 4\n',
                         '[search]\nFULL_SEARCH_VISITS = 4\n'):
                (path / 'run.cfg').write_text('extends = baseline\n' + text)
                with self.assertRaises(ValueError):
                    load_config(path, environ={})

    def test_snapshot_round_trip(self):
        baseline = load_config(ROOT / 'configs/baseline', environ={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for filename in CONFIG_FILES:
                (path / filename).write_text(render_config(baseline, filename))
            self.assertEqual(load_config(path, environ={}), baseline)

    def test_five_file_layout(self):
        expected = {'env.cfg', 'selfplay.cfg', 'net.cfg', 'train.cfg', 'run.cfg'}
        self.assertEqual(set(CONFIG_FILES), expected)
        self.assertEqual({path.name for path in (ROOT / 'configs/baseline').glob('*.cfg')}, expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'run.cfg').write_text('extends = baseline\n')
            (path / 'net.cfg').write_text('[search]\nfull_search_visits = 20\n')
            with self.assertRaises(ValueError):
                load_config(path, environ={})

    def test_parallel_settings_belong_to_train(self):
        _, train = read_profile(ROOT / 'configs/baseline/train.cfg')
        _, selfplay = read_profile(ROOT / 'configs/baseline/selfplay.cfg')
        _, run = read_profile(ROOT / 'configs/baseline/run.cfg')
        parallel = {'TORCH_THREADS', 'NUM_GAME_THREADS', 'NN_MAX_BATCH_SIZE', 'NN_BATCH_WAIT_US'}
        self.assertTrue(parallel <= train.keys())
        self.assertFalse(parallel & (selfplay.keys() | run.keys()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'run.cfg').write_text('extends = baseline\n')
            (path / 'selfplay.cfg').write_text('[parallel]\nnum_game_threads = 2\n')
            with self.assertRaises(ValueError):
                load_config(path, environ={})

    def test_validation(self):
        for key, value in [('NN_MAX_BATCH_SIZE', '0'), ('CHEAP_SEARCH_PROB', 'nan'),
                           ('BOARD_SIZES', '3'), ('UNROLL_STEPS', '-1')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_config(ROOT / 'configs/baseline', environ={key: value})


if __name__ == '__main__':
    unittest.main()
