import tempfile
import unittest
from pathlib import Path

from muzero.config import CONFIG_FILES, ROOT, load_config, read_profile, render_config


class ConfigTests(unittest.TestCase):
    def test_baseline_and_delta(self):
        baseline = load_config(ROOT / 'configs/baseline', environ={})
        experiment = load_config(ROOT / 'configs/exp_baseline', environ={})
        original = load_config(ROOT / 'configs/muzero', environ={})
        self.assertEqual(baseline['BOARD_SIZES'], [15, 14, 13, 12, 11])
        self.assertEqual(experiment['BOARD_SIZES'], [11, 10, 9])
        self.assertEqual(baseline['FULL_SEARCH_VISITS'], 400)
        self.assertEqual(baseline['BOOTSTRAP_EVALUATOR'], 'random')
        self.assertEqual(load_config(ROOT / 'configs/baseline', environ={'BOOTSTRAP_EVALUATOR': 'network'})['BOOTSTRAP_EVALUATOR'], 'network')
        self.assertEqual(experiment['FULL_SEARCH_VISITS'], 220)
        self.assertEqual(baseline['DIRICHLET_TOTAL_CONCENTRATION'], 6.75)
        self.assertEqual(experiment['DIRICHLET_TOTAL_CONCENTRATION'], 3.63)
        self.assertEqual(experiment['VALUE_LOSS_SCALE'], baseline['VALUE_LOSS_SCALE'])
        self.assertEqual(original['VALUE_HEAD'], 'scalar')
        for key in ('AUXILIARY_POLICY_HEADS', 'USE_FPU', 'USE_LCB_FOR_SELECTION',
                    'SHAPED_DIRICHLET_NOISE', 'CHEAP_SEARCH_PROB'):
            self.assertFalse(original[key], key)
        for key in ('SYMMETRY_AUGMENTATION', 'REPLAY_WINDOW', 'SELFPLAY_SCHEDULE'):
            self.assertEqual(original[key], baseline[key])

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
                           ('BOARD_SIZES', '3'), ('UNROLL_STEPS', '-1'), ('BOOTSTRAP_EVALUATOR', 'invalid')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_config(ROOT / 'configs/baseline', environ={key: value})

    def test_network_dimensions_and_depths(self):
        keys = [f'{prefix}_{suffix}' for prefix in (
            'REPRESENTATION', 'DYNAMICS', 'PREDICTION_BACKBONE', 'POLICY_HEAD', 'VALUE_HEAD',
        ) for suffix in ('NUM_BLOCKS', 'NUM_CHANNELS')]
        keys += ['HIDDEN_STATE_NUM_CHANNELS', 'VALUE_HEAD_HIDDEN_CHANNELS']
        for key in keys:
            minimum = 0 if key.endswith('_NUM_BLOCKS') else 1
            with self.subTest(key=key):
                c = load_config(ROOT / 'configs/baseline', environ={key: str(minimum)})
                self.assertEqual(c[key], minimum)
                with self.assertRaisesRegex(ValueError, key):
                    load_config(ROOT / 'configs/baseline', environ={key: str(minimum - 1)})

    def test_legacy_network_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'run.cfg').write_text('extends = baseline\n')
            for key in ('num_blocks', 'num_channels'):
                (path / 'net.cfg').write_text(f'[model]\n{key} = 4\n')
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unknown'):
                    load_config(path, environ={})


if __name__ == '__main__':
    unittest.main()
