import json
import os
from pathlib import Path
import tempfile
import unittest

from muzero.arena import PairStore, Player, execute_pair, file_hash
from muzero.config import load_match_config, render_native_config


class NativeMatchTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('MUZERO_TEST_MATCH_BINARY'), 'Set MUZERO_TEST_MATCH_BINARY and MUZERO_TEST_MODEL_A/B')
    def test_gpu_matches_interrupt_resume_and_color_opening_quotas(self):
        binary = Path(os.environ['MUZERO_TEST_MATCH_BINARY']).resolve()
        paths = [Path(os.environ[f'MUZERO_TEST_MODEL_{side}']).resolve() for side in ('A', 'B')]
        players = tuple(Player(side, side, 2, 1, str(path), file_hash(path)) for side, path in zip(('a', 'b'), paths))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = load_match_config('configs/baseline', environ={
                'MATCH_DEVICE': os.environ.get('MUZERO_TEST_DEVICE', 'cuda:0'),
                'MATCH_FULL_SEARCH_VISITS': '4', 'MATCH_BOARD_SIZE': '5',
                'MATCH_NUM_GAME_THREADS': '1', 'MATCH_NN_BATCH_WAIT_US': '0',
            })
            resolved = output / 'resolved.cfg'
            resolved.write_text(render_native_config(config))
            class InterruptedStore(PairStore):
                interrupted = False

                def accept(self, event):
                    super().accept(event)
                    if event['type'] == 'game' and not self.interrupted:
                        self.interrupted = True
                        raise KeyboardInterrupt

            store = InterruptedStore(output / 'pair', 8)
            with self.assertRaises(KeyboardInterrupt):
                execute_pair(binary, resolved, players, store, 17)
            saved = store.games()
            self.assertGreater(len(saved), 0)
            self.assertLess(len(saved), 8)
            execute_pair(binary, resolved, players, store, 17)
            games = store.games()
            self.assertEqual(len(games), 8)
            self.assertEqual(sum(g['black_a'] for g in games), 4)
            self.assertTrue(all(g in games for g in saved))
            self.assertEqual(store.tasks(17), [])
            for i in range(4):
                opening = json.loads((store.directory / 'openings' / f'{i:08d}.json').read_text())
                self.assertEqual(opening['generator'], i % 2)
                pair = [g for g in games if g['opening_id'] == i]
                self.assertEqual(len(pair), 2)
                for game in pair:
                    self.assertEqual(game['moves'][:len(opening['moves'])], opening['moves'])
                    self.assertEqual(len(set(game['moves'])), len(game['moves']))
                    self.assertTrue(all(0 <= move < 25 for move in game['moves']))
            execute_pair(binary, resolved, players, store, 17)
            self.assertEqual(store.games(), games)


if __name__ == '__main__':
    unittest.main()
