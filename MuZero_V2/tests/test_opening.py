from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from muzero.config import ROOT, load_config
from muzero.protocol import Plane
from muzero.replay import Replay, read_shard, game_statistics
from test_learning import write_game


class OpeningTests(unittest.TestCase):
    def test_configuration_and_validation(self):
        c = load_config(ROOT / 'configs/baseline', environ={})
        self.assertEqual(c['BALANCED_OPENING_PROB'], 0.99)
        self.assertEqual(c['BALANCED_OPENING_MAX_TRIES'], 20)
        self.assertEqual(c['POLICY_INIT_AVG_MOVE_NUM'], 6)
        self.assertEqual(c['POLICY_INIT_TEMPERATURE'], 1.6)
        original = load_config(ROOT / 'configs/muzero', environ={})
        self.assertEqual(original['BALANCED_OPENING_PROB'], 0)
        self.assertFalse(original['INIT_GAMES_WITH_POLICY'])
        for key, value in [('BALANCED_OPENING_PROB', '1.1'), ('BALANCED_OPENING_MAX_TRIES', '0'),
                           ('BALANCED_OPENING_REJECTION_PROB', '-1'), ('POLICY_INIT_TEMPERATURE', '0'),
                           ('POLICY_INIT_AVG_MOVE_NUM', 'nan')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_config(ROOT / 'configs/baseline', environ={key: value})

    def test_opening_prefix_and_white_first_unroll(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5', 'BATCH_SIZE': '1'})
        c['SYMMETRY_AUGMENTATION'] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'shard.mzs'
            write_game(path, opening_count=3)
            record = read_shard(path, 5)[0]
            self.assertEqual(record.opening, (0, 5, 1))
            game = record.load()
            self.assertEqual(len(game), 6)
            self.assertEqual(game[0]['player'], -1)
            self.assertEqual(game[0]['observation'][:2].sum(), 3)
            replay = Replay([record], c)
            for seed in range(20):
                obs, actions, policies, values, masks = replay.sample(np.random.default_rng(seed))
                self.assertGreaterEqual(obs[0, :2].sum(), 3)
                black = bool(obs[0, Plane.BLACK_TO_MOVE, 0, 0])
                np.testing.assert_array_equal(values[0, 0], [1, 0, 0] if black else [0, 0, 1])
            damaged = bytearray(path.read_bytes())
            struct.pack_into('<I', damaged, 44 + 4, 0)
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, 'opening'):
                read_shard(path, 5)[0].load()

    def test_policy_terminal_games_have_no_training_rows(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={'BOARD_SIZES': '5', 'BATCH_SIZE': '1'})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            terminal = directory / 'terminal.mzs'
            normal = directory / 'normal.mzs'
            write_game(terminal, opening_count=9)
            write_game(normal, opening_count=3)
            empty = read_shard(terminal, 5)[0]
            full = read_shard(normal, 5)[0]
            self.assertEqual(empty.rows, 0)
            self.assertEqual(len(empty.load()), 0)
            statistics = game_statistics([empty, full], [empty.load(), full.load()])
            self.assertEqual(statistics['freestyle/5']['games'], 2)
            self.assertEqual(statistics['freestyle/5']['rows'], 6)
            replay = Replay([full, empty], c)
            self.assertEqual(replay.rows, 6)
            self.assertEqual(replay.sample(np.random.default_rng(9))[0].shape[0], 1)


if __name__ == '__main__':
    unittest.main()
