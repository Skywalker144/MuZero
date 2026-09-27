import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from muzero.config import ROOT, load_config, render_native_config
from muzero.protocol import VERSION
from muzero.replay import directory_records, game_statistics


class LinePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.version = VERSION
        self.register_buffer('moves', torch.tensor([0, 5, 1, 6, 2, 7, 3, 8, 4]))

    @torch.jit.export
    def metadata(self):
        return 5, self.version

    @torch.jit.export
    def initial(self, observation: torch.Tensor):
        turns = observation[:, :2].sum(dim=(1, 2, 3)).long().clamp(max=8)
        actions = self.moves[turns]
        logits = observation.new_full((observation.shape[0], 25), -1000)
        logits.scatter_(1, actions.unsqueeze(1), 0)
        return observation, logits, observation.new_zeros((observation.shape[0],))

    @torch.jit.export
    def recurrent(self, hidden: torch.Tensor, actions: torch.Tensor):
        return hidden, hidden.new_zeros((hidden.shape[0], 25)), hidden.new_zeros((hidden.shape[0],))


@unittest.skipUnless(os.environ.get('MUZERO_TEST_BINARY'), 'Requires native selfplay')
class NativeOpeningTests(unittest.TestCase):
    def test_policy_terminal_shards_and_resume(self):
        c = load_config(ROOT / 'configs/minimal_test', environ={
            'DEVICE': os.environ.get('MUZERO_TEST_DEVICE', 'cpu'), 'BOARD_SIZES': '5',
            'RULES': 'freestyle', 'RULE_WEIGHTS': '1',
            'BALANCED_OPENING_PROB': '0', 'POLICY_INIT_AVG_MOVE_NUM': '100',
            'NUM_GAME_THREADS': '2', 'SELFPLAY_ROWS_PER_SHARD': '1',
        })
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = directory / 'line.pt'
            torch.jit.script(LinePolicy()).save(str(model))
            resolved = directory / 'resolved.cfg'
            resolved.write_text(render_native_config(c))
            games = directory / 'games'
            command = [os.environ['MUZERO_TEST_BINARY'], str(resolved), str(model), str(games), '8', '42', '0', 'network']
            subprocess.run(command, check=True, timeout=120)
            records = directory_records(games, 5)
            self.assertEqual(len(records), 8)
            self.assertTrue(any(r.rows == 0 for r in records))
            for record in records:
                self.assertEqual(record.winner, 1)
                self.assertEqual(len(record.opening) + record.rows, 9)
                record.load()
            statistics = game_statistics(records, [r.load() for r in records])
            self.assertEqual(statistics['freestyle/5']['games'], 8)
            before = {p: p.read_bytes() for p in games.glob('*.mzs')}
            subprocess.run(command, check=True, timeout=120)
            self.assertEqual(before, {p: p.read_bytes() for p in games.glob('*.mzs')})
            command[4] = '10'
            subprocess.run(command, check=True, timeout=120)
            self.assertEqual([r.game_id for r in directory_records(games, 5)], list(range(10)))


if __name__ == '__main__':
    unittest.main()
