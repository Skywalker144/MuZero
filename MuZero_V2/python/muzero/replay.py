from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import struct

import numpy as np


class PolicyHead(IntEnum):
    MAIN = 0
    SOFT = 1
    OPPONENT = 2
    SOFT_OPPONENT = 3


class Outcome(IntEnum):
    WIN = 0
    DRAW = 1
    LOSS = 2


def policy_head_count(auxiliary):
    return len(PolicyHead) if auxiliary else 1


@dataclass(frozen=True)
class GameRecord:
    path: Path
    size: int
    winner: int
    rows: int

    @classmethod
    def inspect(cls, path, expected_size):
        path = Path(path)
        with path.open('rb') as stream:
            header = stream.read(20)
        if len(header) != 20:
            raise ValueError(f'Truncated game header: {path}')
        magic, size, winner, rows = struct.unpack('<8sIiI', header)
        if magic != b'MZV2GAME' or size != expected_size or winner not in (-1, 0, 1) or not 1 <= rows <= size * size:
            raise ValueError(f'Invalid game header: {path}')
        if path.stat().st_size != 20 + rows * (16 + 7 * size * size):
            raise ValueError(f'Truncated or oversized game: {path}')
        return cls(path, size, winner, rows)

    def load(self):
        n = self.size * self.size
        dtype = np.dtype([('player', '<i4'), ('action', '<u4'), ('weight', '<f4'),
                          ('visits', '<u4'), ('observation', 'u1', (3, self.size, self.size)),
                          ('policy', '<f4', (n,))])
        data = np.fromfile(self.path, dtype=dtype, offset=20)
        if len(data) != self.rows:
            raise ValueError(f'Game changed after indexing: {self.path}')
        expected_players = np.where(np.arange(self.rows) % 2 == 0, 1, -1)
        if not np.array_equal(data['player'], expected_players):
            raise ValueError(f'Invalid player sequence: {self.path}')
        board = np.zeros(n, dtype=np.int8)
        for step in data:
            player, action = int(step['player']), int(step['action'])
            expected = np.stack([board == player, board == -player, np.full(n, player == 1)])
            if action >= n or board[action] != 0 or not np.array_equal(step['observation'].reshape(3, n), expected):
                raise ValueError(f'Invalid observation/action trajectory: {self.path}')
            board[action] = player
        legal = (data['observation'][:, 0] + data['observation'][:, 1]).reshape(self.rows, n) == 0
        if (not np.isin(data['observation'], [0, 1]).all() or np.any(data['action'] >= n)
                or not legal[np.arange(self.rows), data['action']].all()
                or not np.isin(data['weight'], [0, 1]).all() or (data['visits'] < 1).any()
                or not np.isfinite(data['policy']).all() or (data['policy'] < 0).any()
                or not np.allclose(data['policy'].sum(1), 1, atol=1e-5)
                or (data['policy'][~legal] != 0).any()):
            raise ValueError(f'Invalid game targets: {self.path}')
        return data


def catalog(data_dir, size):
    return [GameRecord.inspect(path, size) for path in sorted(Path(data_dir).glob('selfplay/iter_*/game_*.mzg'))]


def window_size(total, c):
    if c['REPLAY_WINDOW'] == 'fixed':
        return min(c['MAX_ROWS'], c['FIXED_WINDOW_ROWS']) if c['MAX_ROWS'] else c['FIXED_WINDOW_ROWS']
    minimum, exponent = c['MIN_ROWS'], c['TAPER_WINDOW_EXPONENT']
    window = int((total ** exponent - minimum ** exponent) / (exponent * minimum ** (exponent - 1)) * c['EXPAND_WINDOW_PER_ROW'] + minimum)
    window = max(minimum, window)
    return min(c['MAX_ROWS'], window) if c['MAX_ROWS'] else window


class Replay:
    def __init__(self, records, config):
        self.config = config
        self.total = sum(record.rows for record in records)
        target = window_size(self.total, config)
        selected, rows = [], 0
        for record in reversed(records):
            selected.append(record)
            rows += record.rows
            if rows >= target:
                break
        self.records = list(reversed(selected))
        self.games = [record.load() for record in self.records]
        self.ends = np.cumsum([len(game) for game in self.games])
        self.rows = rows

    def sample(self, rng):
        c = self.config
        n, k, b = c['BOARD_SIZE'] ** 2, c['UNROLL_STEPS'], c['BATCH_SIZE']
        if self.rows < b:
            raise ValueError('Replay has fewer rows than BATCH_SIZE')
        observations = np.empty((b, 3, c['BOARD_SIZE'], c['BOARD_SIZE']), dtype=np.float32)
        actions = np.zeros((b, k), dtype=np.int64)
        heads = policy_head_count(c['AUXILIARY_POLICY_HEADS'])
        policies = np.zeros((b, k + 1, heads, n), dtype=np.float32)
        values = np.zeros((b, k + 1, len(Outcome)), dtype=np.float32)
        masks = np.zeros((b, k + 1, heads), dtype=np.float32)
        for batch, position in enumerate(rng.integers(self.rows, size=b)):
            index = int(np.searchsorted(self.ends, position, side='right'))
            start = int(position - (self.ends[index - 1] if index else 0))
            game, record = self.games[index], self.records[index]
            observations[batch] = game[start]['observation']
            for step in range(k + 1):
                row = start + step
                player = int(game[start]['player']) * (-1 if step % 2 else 1)
                outcome = record.winner * player
                values[batch, step, Outcome.WIN if outcome > 0 else Outcome.LOSS if outcome < 0 else Outcome.DRAW] = 1
                if row >= len(game):
                    continue
                if step < k:
                    actions[batch, step] = game[row]['action']
                if not c['AUXILIARY_POLICY_HEADS']:
                    policies[batch, step, PolicyHead.MAIN] = game[row]['policy']
                    masks[batch, step, PolicyHead.MAIN] = game[row]['weight']
                    continue
                for offset, main, soft in ((0, PolicyHead.MAIN, PolicyHead.SOFT), (1, PolicyHead.OPPONENT, PolicyHead.SOFT_OPPONENT)):
                    target_row = row + offset
                    if target_row >= len(game):
                        continue
                    target = game[target_row]
                    legal = (target['observation'][0] + target['observation'][1]).reshape(n) == 0
                    softened = np.zeros(n, dtype=np.float32)
                    softened[legal] = (target['policy'][legal] + c['SOFT_POLICY_EPS']) ** (1 / c['SOFT_POLICY_TEMPERATURE'])
                    policies[batch, step, main] = target['policy']
                    policies[batch, step, soft] = softened / softened.sum()
                    masks[batch, step, [main, soft]] = target['weight']
        if c['SYMMETRY_AUGMENTATION']:
            observations, actions, policies = augment(observations, actions, policies, rng)
        return observations, actions, policies, values, masks


def augment(observations, actions, policies, rng):
    size = observations.shape[-1]
    observations, actions, policies = observations.copy(), actions.copy(), policies.copy()
    for i in range(len(observations)):
        rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
        transform = lambda plane: np.flip(np.rot90(plane, rotation, axes=(-2, -1)), axis=-1) if flip else np.rot90(plane, rotation, axes=(-2, -1))
        indices = transform(np.arange(size * size).reshape(size, size)).reshape(-1)
        inverse = np.argsort(indices)
        observations[i] = transform(observations[i])
        actions[i] = inverse[actions[i]]
        policies[i] = policies[i][..., indices]
    return observations, actions, policies
