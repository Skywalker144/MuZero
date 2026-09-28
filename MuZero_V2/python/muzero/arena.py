import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys
from threading import Thread

from .config import ROOT, load_match_config, render_native_config
from .process import install_signals, stop_process
from .storage import run_lock, write_json


@dataclass(frozen=True)
class Player:
    id: str
    arm: str
    iteration: int
    seconds: float
    model: str
    sha256: str


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['games_per_pair'] > manifest['games_per_pair'] or (
                previous | {'games_per_pair': manifest['games_per_pair']}) != manifest:
            raise ValueError('Models, evaluator, binary or schedule changed; use a separate output directory')
    write_json(path, manifest)


def discover_players(root: Path, stride: int) -> list[Player]:
    if stride < 1:
        raise ValueError('Stride must be positive')
    players = []
    for arm in sorted(root.iterdir()):
        state_path = arm / 'logs/state.json'
        if not state_path.is_file():
            continue
        committed = json.loads(state_path.read_text())['iteration']
        rows = []
        for iteration in range(committed):
            path = arm / 'logs/iters' / f'{iteration:08d}.json'
            if not path.is_file():
                raise ValueError(f'Missing committed iteration: {path}')
            row = json.loads(path.read_text())
            seconds = row['elapsed_seconds']
            if row['iteration'] != iteration or not math.isfinite(seconds) or seconds < 0 or (rows and seconds < rows[-1]['elapsed_seconds']):
                raise ValueError(f'Invalid iteration timing: {path}')
            rows.append(row)
        available = sorted((arm / 'models').glob('model_*.pt'))
        first = next((int(p.stem.removeprefix('model_')) for p in available
                      if 1 <= int(p.stem.removeprefix('model_')) <= committed), None)
        if first is None:
            continue
        for row in rows:
            number = row['iteration'] + 1
            if number < first or (number not in (first, committed) and number % stride):
                continue
            model = arm / 'models' / f'model_{number:08d}.pt'
            if not model.is_file():
                raise ValueError(f'Missing selected model: {model}')
            players.append(Player(f'{arm.name}:{number:08d}', arm.name, number, row['elapsed_seconds'],
                                  str(model.resolve()), file_hash(model)))
    if len(players) < 2:
        raise ValueError('Need at least two committed models')
    return players


def build_schedule(players: list[Player], neighbors: int) -> list[tuple[str, str]]:
    if neighbors < 1:
        raise ValueError('Neighbors must be positive')
    arms = {}
    for player in players:
        arms.setdefault(player.arm, []).append(player)
    edges = set()
    for rows in arms.values():
        rows.sort(key=lambda p: p.iteration)
        for i, player in enumerate(rows):
            for other in rows[max(0, i-neighbors):i]:
                edges.add(tuple(sorted((player.id, other.id))))
    for a, b in combinations(arms.values(), 2):
        horizon = min(a[-1].seconds, b[-1].seconds)
        for source, target in ((a, b), (b, a)):
            for player in source:
                if player.seconds > horizon:
                    continue
                other = min(target, key=lambda p: abs(p.seconds - player.seconds))
                edges.add(tuple(sorted((player.id, other.id))))
    return sorted(edges)


class PairStore:
    def __init__(self, directory: Path, games: int):
        if games < 4 or games % 4:
            raise ValueError('Games per pair must be a positive multiple of 4')
        self.directory = directory
        self.total = games

    def accept(self, event: dict) -> None:
        kind, index = event['type'], event['id']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid event ID')
        moves = event['moves']
        if not isinstance(moves, list) or not moves or any(type(m) is not int or m < 0 for m in moves) or len(set(moves)) != len(moves):
            raise ValueError('Invalid move history')
        if kind == 'opening':
            if index >= self.total // 2 or event['generator'] != index % 2 or not math.isfinite(event['value']):
                raise ValueError('Invalid opening assignment')
            folder = 'openings'
        elif kind == 'game':
            if index >= self.total or event['opening_id'] != index // 2 or event['black_a'] is not (index % 2 == 0) or event['winner'] not in (-1, 0, 1):
                raise ValueError('Invalid game assignment')
            opening = json.loads((self.directory / 'openings' / f'{index//2:08d}.json').read_text())
            if moves[:len(opening['moves'])] != opening['moves'] or len(moves) <= len(opening['moves']):
                raise ValueError('Game does not continue its opening')
            if not math.isfinite(event['seconds']) or event['seconds'] < 0:
                raise ValueError('Invalid game duration')
            folder = 'games'
        else:
            raise ValueError(f'Unknown match event: {kind}')
        path = self.directory / folder / f'{index:08d}.json'
        if path.exists():
            if json.loads(path.read_text()) != event:
                raise ValueError(f'Conflicting completed event: {path}')
        else:
            write_json(path, event)

    def tasks(self, seed: int) -> list[dict]:
        tasks = []
        for index in range(self.total // 2):
            mask = sum(1 << color for color in (0, 1)
                       if not (self.directory / 'games' / f'{2*index+color:08d}.json').exists())
            if not mask:
                continue
            opening_path = self.directory / 'openings' / f'{index:08d}.json'
            opening_seed = (seed + index * 0x9e3779b97f4a7c15) % (1 << 64)
            moves = None
            if opening_path.exists():
                opening = json.loads(opening_path.read_text())
                self.accept(opening)
                if opening['seed'] != opening_seed or opening['generator'] != index % 2:
                    raise ValueError('Saved opening does not match scheduled seed/generator')
                moves = opening['moves']
            tasks.append({'id': index, 'seed': opening_seed, 'generator': index % 2, 'mask': mask, 'moves': moves})
        return tasks

    def games(self) -> list[dict]:
        rows = []
        for path in sorted((self.directory / 'games').glob('*.json')):
            row = json.loads(path.read_text())
            if row['type'] != 'game' or path.name != f"{row['id']:08d}.json":
                raise ValueError(f'Invalid result identity: {path}')
            self.accept(row)
            rows.append(row)
        return rows


def execute_pair(binary: Path, config: Path, players: tuple[Player, Player], store: PairStore, seed: int) -> None:
    completed = len(store.games())
    tasks = store.tasks(seed)
    if not tasks:
        return
    task_path = store.directory / 'tasks.txt'
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(''.join(
        f"{t['id']} {t['seed']} {t['generator']} {t['mask']} " +
        ('-1' if t['moves'] is None else f"{len(t['moves'])} " + ' '.join(map(str, t['moves']))) + '\n'
        for t in tasks))
    with (store.directory / 'native.log').open('a') as log:
        process = subprocess.Popen([str(binary), str(config), *(p.model for p in players), str(task_path)],
                                   stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True)
        try:
            for line in process.stdout:
                event = json.loads(line)
                store.accept(event)
                if event['type'] == 'game':
                    completed += 1
                    print(f"  games={completed}/{store.total}", flush=True)
            code = process.wait()
            if code:
                raise RuntimeError(f'Match exited {code}: {store.directory / "native.log"}')
        finally:
            stopper = Thread(target=stop_process, args=(process,))
            stopper.start()
            try:
                for line in process.stdout:
                    store.accept(json.loads(line))
            finally:
                process.stdout.close()
                stopper.join()
    if store.tasks(seed):
        raise RuntimeError('Match exited without completing its assignments')


def load_results(output: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((output / 'manifest.json').read_text())
    games = []
    for pair in manifest['pairs']:
        a, b = pair['players']
        store = PairStore(output / 'pairs' / pair['id'], manifest['games_per_pair'])
        for row in store.games():
            games.append({'pair': pair['id'], 'opening_id': row['opening_id'],
                          'black': a if row['black_a'] else b, 'white': b if row['black_a'] else a,
                          'score': (row['winner'] + 1) / 2})
    return manifest, games


def main():
    parser = argparse.ArgumentParser(description='Resumable native matches and equal-wall-time Elo')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/exp_muzero_opt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config-dir', default='configs/baseline')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/muzero_match')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--neighbors', type=int, default=2)
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--anchor', default='')
    parser.add_argument('--bootstrap-samples', type=int, default=300)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    output = (args.output or args.data / 'elo').resolve()
    if args.fit_only:
        from .elo import write_ratings
        write_ratings(output, args.bootstrap_samples)
        return
    config = load_match_config(args.config_dir)
    binary = args.binary.resolve()
    players = discover_players(args.data.resolve(), args.stride)
    schedule = build_schedule(players, args.neighbors)
    anchor = args.anchor
    if not anchor:
        baseline = [p for p in players if p.arm == 'exp_baseline'] or [p for p in players if p.arm == players[0].arm]
        anchor = min(baseline, key=lambda p: abs(p.seconds - baseline[-1].seconds / 2)).id
    if anchor not in {p.id for p in players}:
        raise ValueError(f'Unknown anchor: {anchor}')
    PairStore(output, args.games)
    manifest = {'version': 1, 'players': [asdict(p) for p in players], 'config': config,
                'binary_sha256': file_hash(binary), 'games_per_pair': args.games, 'anchor': anchor,
                'pairs': [{'id': identity(pair), 'players': list(pair)} for pair in schedule]}
    print(f'players={len(players)} pairs={len(schedule)} games={len(schedule)*args.games} anchor={anchor}', flush=True)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    install_signals()
    with run_lock(output / 'arena.lock'):
        path = output / 'manifest.json'
        save_manifest(path, manifest)
        config_path = output / 'resolved.cfg'
        config_path.write_text(render_native_config(config))
        by_id = {p.id: p for p in players}
        for index, pair in enumerate(manifest['pairs']):
            print(f"pair={index+1}/{len(schedule)} {' vs '.join(pair['players'])}", flush=True)
            store = PairStore(output / 'pairs' / pair['id'], args.games)
            seed = (config['SEED'] + int(pair['id'][:16], 16)) % (1 << 64)
            execute_pair(binary, config_path, tuple(by_id[p] for p in pair['players']), store, seed)
        from .elo import write_ratings
        write_ratings(output, args.bootstrap_samples)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
