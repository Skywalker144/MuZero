import json
from pathlib import Path
import tempfile
import unittest

from muzero.config import ROOT, load_config
from muzero.experiment import experiment_plan
from muzero.run import plan_games
from muzero.storage import atomic_path, run_lock, write_json


class PipelineTests(unittest.TestCase):
    def test_zero_row_iteration_collects_another_bootstrap_budget(self):
        c = load_config(ROOT / 'configs/baseline', environ={})
        for iteration in (1, 2, 3):
            state = {'iteration': iteration, 'target_rows': 0, 'rows_per_game': 0}
            self.assertEqual(plan_games(state, c, 0), c['BOOTSTRAP_GAMES'])

    def test_schedule_bootstrap_and_backfill(self):
        c = load_config(ROOT / 'configs/baseline', environ={'MIN_ROWS': '30000', 'BACKFILL_FACTOR': '1.1'})
        state = {'iteration': 0, 'target_rows': 0, 'rows_per_game': 50}
        self.assertEqual(plan_games(state, c, 0), c['BOOTSTRAP_GAMES'])
        state['iteration'] = 1
        self.assertEqual(plan_games(state, c, 10000), 440)
        self.assertEqual(state['target_rows'], 0)
        self.assertEqual(plan_games(state, c, 40000), 0)

    def test_iteration_two_starts_a_full_budget_at_actual_bootstrap_rows(self):
        c = load_config(ROOT / 'configs/baseline', environ={
            'MIN_ROWS': '30000', 'TRAIN_STEPS': '200', 'BATCH_SIZE': '128', 'REPLAY_RATIO': '8',
        })
        for total_rows in (29000, 30000, 40000):
            with self.subTest(total_rows=total_rows):
                state = {'iteration': 2, 'target_rows': 0, 'rows_per_game': 50}
                self.assertEqual(plan_games(state, c, total_rows), 64)
                self.assertEqual(state['target_rows'], total_rows + 3200)

    def test_normal_schedule_carries_credit_and_debt_using_previous_average(self):
        c = load_config(ROOT / 'configs/baseline', environ={
            'TRAIN_STEPS': '200', 'BATCH_SIZE': '128', 'REPLAY_RATIO': '8',
        })
        state = {'iteration': 3, 'target_rows': 43200, 'rows_per_game': 62.5}
        self.assertEqual(plan_games(state, c, 44000), 39)
        self.assertEqual(state['target_rows'], 46400)
        state.update(iteration=4, rows_per_game=50)
        self.assertEqual(plan_games(state, c, 45950), 73)
        self.assertEqual(state['target_rows'], 49600)

    def test_normal_schedule_always_collects_and_preserves_excess_credit(self):
        c = load_config(ROOT / 'configs/baseline', environ={
            'TRAIN_STEPS': '200', 'BATCH_SIZE': '128', 'REPLAY_RATIO': '8',
        })
        state = {'iteration': 3, 'target_rows': 43200, 'rows_per_game': 50}
        self.assertEqual(plan_games(state, c, 50000), 1)
        self.assertEqual(state['target_rows'], 46400)
        state['iteration'] = 4
        self.assertEqual(plan_games(state, c, 50050), 1)
        self.assertEqual(state['target_rows'], 49600)
        state['iteration'] = 5
        self.assertEqual(plan_games(state, c, 50100), 54)
        self.assertEqual(state['target_rows'], 52800)

    def test_fixed_collection_has_no_cumulative_adjustment(self):
        c = load_config(ROOT / 'configs/exp_muzero', environ={'SELFPLAY_SCHEDULE': 'fixed'})
        for iteration in range(4):
            with self.subTest(iteration=iteration):
                state = {'iteration': iteration, 'target_rows': 0, 'rows_per_game': 50}
                self.assertEqual(plan_games(state, c, 100000), c['GAMES_PER_ITER'])
                self.assertEqual(state['target_rows'], 0)

    def test_atomic_failure_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.json'
            write_json(path, {'iteration': 1})
            with self.assertRaises(RuntimeError):
                with atomic_path(path) as temporary:
                    temporary.write_text('partial')
                    raise RuntimeError('interruption')
            self.assertEqual(json.loads(path.read_text()), {'iteration': 1})
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_exclusive_run_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(Path(tmp) / 'lock'):
                with self.assertRaises(RuntimeError):
                    with run_lock(Path(tmp) / 'lock'):
                        pass

    def test_experiment_limits_and_destinations(self):
        _, settings, arms, slots = experiment_plan(ROOT / 'configs/exp_muzero_opt', environ={})
        self.assertEqual(len(arms), 2)
        self.assertEqual([arm['name'] for arm in arms], ['exp_baseline', 'exp_muzero'])
        for arm in arms:
            expected = load_config(ROOT / 'configs' / arm['name'], environ={})
            actual = arm['config']
            for key in expected.keys() - {'DATA_DIR', 'MAX_ITERS', 'MAX_TIME_SECONDS'}:
                self.assertEqual(actual[key], expected[key], (arm['name'], key))
            self.assertEqual(actual['MAX_TIME_SECONDS'], 18000)
        self.assertEqual(settings['MAX_TIME_SECONDS'], 18000)
        self.assertEqual(load_config(ROOT / 'configs/baseline', environ={})['MAX_TIME_SECONDS'], 0)
        self.assertEqual(slots, [None])
        with self.assertRaises(ValueError):
            experiment_plan(ROOT / 'configs/exp_muzero_opt', environ={'DATA_DIR': '/tmp/shared-data'})
        with self.assertRaises(ValueError):
            experiment_plan(ROOT / 'configs/exp_muzero_opt', environ={'MAX_TIME_SECONDS': '0', 'MAX_ITERS': '0'})

    def test_experiment_file_sets_limits_per_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'exp.cfg').write_text('[experiment]\nmax_time_seconds = 120\nmax_iters = 3\nshared_init = true\narm_gpus =\n')
            for name in ('baseline', 'muzero'):
                (directory / name).mkdir()
                (directory / name / 'run.cfg').write_text(f'extends = {name}\n')
            _, settings, arms, _ = experiment_plan(directory, environ={})
            self.assertEqual(settings['MAX_TIME_SECONDS'], 120)
            self.assertTrue(all(arm['config']['MAX_TIME_SECONDS'] == 120 for arm in arms))
            _, _, arms, _ = experiment_plan(directory, environ={'MAX_TIME_SECONDS': '60'})
            self.assertTrue(all(arm['config']['MAX_TIME_SECONDS'] == 60 for arm in arms))

    def test_experiment_requires_its_own_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'exp.cfg').write_text('[experiment]\nmax_time_seconds = 120\nmax_iters = 3\n')
            with self.assertRaisesRegex(ValueError, 'Missing experiment keys.*ARM_GPUS.*SHARED_INIT'):
                experiment_plan(directory, environ={})

    def test_run_config_rejects_experiment_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'run.cfg').write_text('extends = baseline\n[experiment]\nmax_time_seconds = 120\n')
            with self.assertRaises(ValueError):
                load_config(directory, environ={})


if __name__ == '__main__':
    unittest.main()
