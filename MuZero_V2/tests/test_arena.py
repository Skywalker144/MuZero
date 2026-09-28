import json
from pathlib import Path
import tempfile
import unittest

from muzero.arena import Player, build_schedule, discover_players, PairStore, save_manifest
from muzero.config import load_match_config, load_config


class ArenaTests(unittest.TestCase):
    def test_resume_can_add_games_but_cannot_mix_evaluators(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            manifest = {'games_per_pair': 40, 'config': {'FULL_SEARCH_VISITS': 200}, 'pairs': []}
            save_manifest(path, manifest)
            save_manifest(path, dict(manifest, games_per_pair=80))
            self.assertEqual(json.loads(path.read_text())['games_per_pair'], 80)
            for changed in (manifest, dict(manifest, games_per_pair=80, config={'FULL_SEARCH_VISITS': 100})):
                with self.assertRaises(ValueError):
                    save_manifest(path, changed)

    def test_match_config_is_independent_and_validated(self):
        c = load_match_config('configs/exp_muzero', environ={'FULL_SEARCH_VISITS': '3'})
        self.assertEqual(c['FULL_SEARCH_VISITS'], 200)
        self.assertEqual(c['NUM_SEARCH_THREADS'], 1)
        self.assertEqual(c['BALANCED_OPENING_BALANCE_EXPONENT'], 10)
        self.assertEqual(c['BOARD_SIZE'], 15)
        self.assertEqual(c['RULE'], 'renju')
        self.assertTrue(c['USE_LCB_FOR_SELECTION'])
        self.assertNotIn('CHEAP_SEARCH_PROB', c)
        for key, value in [('BOARD_SIZE', '4'), ('RULE', 'bad'), ('NUM_GAME_THREADS', '0'),
                           ('BALANCED_OPENING_PROB', '0.9')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_match_config('configs/baseline', environ={f'MATCH_{key}': value})
        self.assertEqual(load_config('configs/baseline', environ={})['BALANCED_OPENING_BALANCE_EXPONENT'], 4)

    def test_selection_uses_committed_models_and_measured_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm = root / 'a'
            (arm / 'logs/iters').mkdir(parents=True)
            (arm / 'models').mkdir()
            (arm / 'logs/state.json').write_text(json.dumps({'iteration': 9}))
            for i in range(1, 11):
                (arm / 'logs/iters' / f'{i-1:08d}.json').write_text(json.dumps({
                    'iteration': i-1, 'elapsed_seconds': i*i,
                }))
                if i > 1:
                    (arm / 'models' / f'model_{i:08d}.pt').write_bytes(str(i).encode())
            players = discover_players(root, 4)
            self.assertEqual([p.iteration for p in players], [2, 4, 8, 9])
            self.assertEqual([p.seconds for p in players], [4, 16, 64, 81])
            (arm / 'models/model_00000004.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                discover_players(root, 4)

    def test_schedule_has_skip_edges_and_cross_arm_bridges(self):
        players = [Player(f'{arm}:{i}', arm, i, i * speed, '/model', f'{arm}{i}')
                   for arm, speed in [('a', 10), ('b', 20), ('c', 15)] for i in [2, 4, 8]]
        pairs = build_schedule(players, 2)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertIn(('a:2', 'a:8'), pairs)
        self.assertIn(('a:4', 'b:2'), pairs)
        reachable = {players[0].id}
        while True:
            expanded = reachable | {p for pair in pairs if reachable.intersection(pair) for p in pair}
            if expanded == reachable:
                break
            reachable = expanded
        self.assertEqual(reachable, {p.id for p in players})

    def test_store_preserves_opening_and_exact_remaining_color(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PairStore(Path(tmp), 40)
            opening = {'type': 'opening', 'id': 0, 'generator': 0, 'seed': 71,
                       'moves': [112], 'attempts': 1, 'value': 0.1}
            store.accept(opening)
            game = {'type': 'game', 'id': 0, 'opening_id': 0, 'black_a': True,
                    'winner': 1, 'moves': [112, 0], 'seconds': 0.2}
            store.accept(game)
            store.accept(game)
            tasks = store.tasks(71)
            self.assertEqual(tasks[0]['mask'], 2)
            self.assertEqual(tasks[0]['moves'], [112])
            self.assertEqual(len(store.games()), 1)
            with self.assertRaises(ValueError):
                store.accept(dict(game, winner=-1))
            with self.assertRaises(ValueError):
                store.accept(dict(opening, id=1, generator=0))
            with self.assertRaises(ValueError):
                PairStore(Path(tmp), 42)
            restarted = PairStore(Path(tmp), 40)
            self.assertEqual(restarted.tasks(71), tasks)


if __name__ == '__main__':
    unittest.main()
