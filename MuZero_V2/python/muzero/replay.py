from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import json
import struct
import zlib

import numpy as np

from .storage import write_json

from .protocol import INPUT_PLANES, MAGIC, Plane, Rule


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


def placement_mask(observations):
    return (observations[..., Plane.ON_BOARD, :, :] != 0) & (
        observations[..., Plane.OWN, :, :] + observations[..., Plane.OPPONENT, :, :] == 0)


@dataclass(frozen=True)
class GameRecord:
    path: Path
    canvas: int
    size: int
    rule: Rule
    winner: int
    rows: int
    game_id: int
    offset: int
    compressed_bytes: int
    opening: tuple[int, ...]

    def load(self):
        n = self.canvas * self.canvas
        packed = (INPUT_PLANES * n + 7) // 8
        fields = [('player', '<i4'), ('action', '<u4'), ('weight', '<f4'), ('visits', '<u4')]
        disk_dtype = np.dtype(fields + [('packed', 'u1', (packed,)), ('policy', '<f4', (n,))])
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            compressed = stream.read(self.compressed_bytes)
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed, self.rows * disk_dtype.itemsize + 1)
        except zlib.error as error:
            raise ValueError(f'Corrupt game: {self.path}:{self.game_id}') from error
        if len(raw) != self.rows * disk_dtype.itemsize or not decoder.eof or decoder.unused_data:
            raise ValueError(f'Invalid compressed game: {self.path}:{self.game_id}')
        stored = np.frombuffer(raw, dtype=disk_dtype)
        dtype = np.dtype(fields + [('observation', 'u1', (INPUT_PLANES, self.canvas, self.canvas)),
                                   ('policy', '<f4', (n,))])
        data = np.empty(self.rows, dtype=dtype)
        for name in ('player', 'action', 'weight', 'visits', 'policy'):
            data[name] = stored[name]
        data['observation'] = np.unpackbits(stored['packed'], axis=1, count=INPUT_PLANES * n).reshape(
            self.rows, INPUT_PLANES, self.canvas, self.canvas)
        expected_players = np.where((np.arange(self.rows) + len(self.opening)) % 2 == 0, 1, -1)
        if not np.array_equal(data['player'], expected_players):
            raise ValueError(f'Invalid player sequence: {self.path}')
        board = np.zeros((self.canvas, self.canvas), dtype=np.int8)
        on_board = np.zeros_like(board, dtype=bool)
        on_board[:self.size, :self.size] = True
        for turn, action in enumerate(self.opening):
            if not 0 <= action < n or not on_board.flat[action] or board.flat[action] != 0:
                raise ValueError(f'Invalid opening trajectory: {self.path}')
            board.flat[action] = 1 if turn % 2 == 0 else -1
        for step in data:
            player, action = int(step['player']), int(step['action'])
            obs = step['observation']
            if (action >= n or not placement_mask(obs).reshape(n)[action]
                    or not np.array_equal(obs[Plane.OWN], board == player)
                    or not np.array_equal(obs[Plane.OPPONENT], board == -player)
                    or not np.array_equal(obs[Plane.ON_BOARD], on_board)
                    or not np.array_equal(obs[Plane.BLACK_TO_MOVE], on_board & (player == 1))
                    or not np.array_equal(obs[Plane.STANDARD], on_board & (self.rule == Rule.STANDARD))
                    or not np.array_equal(obs[Plane.RENJU], on_board & (self.rule == Rule.RENJU))):
                raise ValueError(f'Invalid observation/action trajectory: {self.path}')
            for plane, active_player in ((Plane.FORBIDDEN_BLACK_TURN, 1), (Plane.FORBIDDEN_WHITE_TURN, -1)):
                forbidden = obs[plane] != 0
                if (forbidden & ~((board == 0) & on_board)).any() or (
                    (self.rule != Rule.RENJU or player != active_player) and forbidden.any()
                ):
                    raise ValueError(f'Invalid forbidden-point feature: {self.path}')
            board.flat[action] = player
        legal = placement_mask(data['observation']).reshape(self.rows, n)
        if (not np.isin(data['observation'], [0, 1]).all()
                or not np.isfinite(data['weight']).all() or (data['weight'] < 0).any() or (data['visits'] < 1).any()
                or not np.isfinite(data['policy']).all() or (data['policy'] < 0).any()
                or not np.allclose(data['policy'].sum(1), 1, atol=1e-5)
                or (data['policy'][~legal] != 0).any()):
            raise ValueError(f'Invalid game targets: {self.path}')
        return data


def read_shard(path, canvas):
    path = Path(path)
    records = []
    size_on_disk = path.stat().st_size
    with path.open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f'Truncated shard: {path}')
        magic, count = struct.unpack('<8sI', header)
        if magic != MAGIC or not 1 <= count <= size_on_disk // 32:
            raise ValueError(f'Invalid shard header: {path}')
        for _ in range(count):
            header = stream.read(32)
            if len(header) != 32:
                raise ValueError(f'Truncated game index: {path}')
            game_id, stored_canvas, size, rule, winner, rows, opening_count, compressed = struct.unpack('<IIIIiIII', header)
            if (stored_canvas != canvas or not 5 <= size <= canvas <= 25
                    or rule not in [int(value) for value in Rule] or winner not in (-1, 0, 1)
                    or not 1 <= rows + opening_count <= size * size or compressed < 1
                    or stream.tell() + 4 * opening_count + compressed > size_on_disk):
                raise ValueError(f'Invalid game index: {path}')
            opening = struct.unpack(f'<{opening_count}I', stream.read(4 * opening_count))
            records.append(GameRecord(path, canvas, size, Rule(rule), winner, rows,
                                      game_id, stream.tell(), compressed, opening))
            stream.seek(compressed, 1)
        if stream.tell() != size_on_disk or len({r.game_id for r in records}) != count:
            raise ValueError(f'Trailing bytes or duplicate game IDs: {path}')
    return records


def directory_records(directory, canvas):
    records = [record for path in sorted(Path(directory).glob('shard_*.mzs')) for record in read_shard(path, canvas)]
    if len({r.game_id for r in records}) != len(records):
        raise ValueError(f'Duplicate game IDs: {directory}')
    return sorted(records, key=lambda record: record.game_id)


def catalog(data_dir, canvas):
    return [record for directory in sorted(Path(data_dir).glob('selfplay/iter_*'))
            for record in directory_records(directory, canvas)]


def game_statistics(records, games):
    groups = {}
    for record, game in zip(records, games):
        key = f'{record.rule.name.lower()}/{record.size}'
        group = groups.setdefault(key, dict(games=0, rows=0, black_wins=0, white_wins=0, draws=0, forbidden_losses=0))
        group['games'] += 1
        group['rows'] += record.rows
        group['black_wins'] += int(record.winner == 1)
        group['white_wins'] += int(record.winner == -1)
        group['draws'] += int(record.winner == 0)
        if not len(game):
            group['forbidden_losses'] += int(record.rule == Rule.RENJU and record.winner == -1
                                            and len(record.opening) % 2 == 1)
            continue
        last = game[-1]
        group['forbidden_losses'] += int(record.rule == Rule.RENJU and record.winner == -1 and last['player'] == 1
            and last['observation'][Plane.FORBIDDEN_BLACK_TURN].reshape(-1)[last['action']] != 0)
    return groups


def window_size(total, c):
    if c['REPLAY_WINDOW'] == 'fixed':
        return min(c['MAX_ROWS'], c['FIXED_WINDOW_ROWS']) if c['MAX_ROWS'] else c['FIXED_WINDOW_ROWS']
    minimum, exponent = c['MIN_ROWS'], c['TAPER_WINDOW_EXPONENT']
    window = int((total ** exponent - minimum ** exponent) / (exponent * minimum ** (exponent - 1)) * c['EXPAND_WINDOW_PER_ROW'] + minimum)
    window = max(minimum, window)
    return min(c['MAX_ROWS'], window) if c['MAX_ROWS'] else window


class Replay:
    def __init__(self, records, config, snapshot_path=None):
        self.config = config
        self.total = sum(record.rows for record in records)
        target = window_size(self.total, config)
        selected, rows = [], 0
        for record in reversed(records):
            if record.rows == 0:
                continue
            selected.append(record)
            rows += record.rows
            if rows >= target:
                break
        self.records = list(reversed(selected))
        if any(record.canvas != config['CANVAS_SIZE'] for record in self.records):
            raise ValueError('Replay canvas mismatch')
        if snapshot_path is not None:
            snapshot_path = Path(snapshot_path)
            snapshot = {'total_rows': self.total, 'rows': rows, 'games': [
                {'path': str(record.path.relative_to(snapshot_path.parent.parent)), 'id': record.game_id,
                 'rows': record.rows} for record in self.records]}
            if snapshot_path.exists():
                if json.loads(snapshot_path.read_text()) != snapshot:
                    raise ValueError('Replay snapshot changed during an unfinished iteration')
            else:
                write_json(snapshot_path, snapshot)
        self.games = [record.load() for record in self.records]
        self.ends = np.cumsum([len(game) for game in self.games])
        self.rows = rows
        row_weights = np.concatenate([game['weight'] for game in self.games]) if self.games else np.empty(0)
        self.weight_ends = np.cumsum(row_weights, dtype=np.float64)
        self.weight_sum = float(self.weight_ends[-1]) if len(self.weight_ends) else 0.0

    def sample(self, rng):
        c = self.config
        n, k, b = c['CANVAS_SIZE'] ** 2, c['UNROLL_STEPS'], c['BATCH_SIZE']
        if self.rows < b:
            raise ValueError('Replay has fewer rows than BATCH_SIZE')
        observations = np.empty((b, INPUT_PLANES, c['CANVAS_SIZE'], c['CANVAS_SIZE']), dtype=np.float32)
        actions = np.zeros((b, k), dtype=np.int64)
        heads = policy_head_count(c['AUXILIARY_POLICY_HEADS'])
        policies = np.zeros((b, k + 1, heads, n), dtype=np.float32)
        values = np.zeros((b, k + 1, len(Outcome)), dtype=np.float32)
        masks = np.zeros((b, k + 1, heads), dtype=np.float32)
        weights = np.ones((b, k + 1), dtype=np.float32)
        if self.weight_sum <= 0:
            raise ValueError('Replay has no positive training weights')
        positions = np.searchsorted(self.weight_ends, rng.random(b) * self.weight_sum, side='right')
        for batch, position in enumerate(positions):
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
                if step > 0:
                    weights[batch, step] = game[row]['weight'] * self.rows / self.weight_sum
                if step < k:
                    actions[batch, step] = game[row]['action']
                if not c['AUXILIARY_POLICY_HEADS']:
                    policies[batch, step, PolicyHead.MAIN] = game[row]['policy']
                    masks[batch, step, PolicyHead.MAIN] = 1
                    continue
                for offset, main, soft in ((0, PolicyHead.MAIN, PolicyHead.SOFT), (1, PolicyHead.OPPONENT, PolicyHead.SOFT_OPPONENT)):
                    target_row = row + offset
                    if target_row >= len(game):
                        continue
                    target = game[target_row]
                    legal = target['observation'][Plane.ON_BOARD].reshape(n) != 0
                    softened = np.zeros(n, dtype=np.float32)
                    softened[legal] = (target['policy'][legal] + c['SOFT_POLICY_EPS']) ** (1 / c['SOFT_POLICY_TEMPERATURE'])
                    policies[batch, step, main] = target['policy']
                    policies[batch, step, soft] = softened / softened.sum()
                    masks[batch, step, [main, soft]] = 1
        if c['SYMMETRY_AUGMENTATION']:
            observations, actions, policies = augment(observations, actions, policies, rng)
        return observations, actions, policies, values, masks, weights


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
