import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from .protocol import VERSION
from .config import ROOT, load_config, model_identity, render_config, render_native_config
from .process import install_signals, run_process
from .storage import atomic_path, run_lock, write_json

MUTABLE_KEYS = set('DEVICE TORCH_THREADS NUM_GAME_THREADS NN_MAX_BATCH_SIZE NN_BATCH_WAIT_US MAX_ITERS MAX_TIME_SECONDS DATA_DIR INIT_MODEL'.split())


def build_binary():
    if os.environ.get('MUZERO_BINARY'):
        binary = Path(os.environ['MUZERO_BINARY']).resolve()
    else:
        with run_lock(ROOT / 'build/build.lock'):
            run_process(['bash', str(ROOT / 'scripts/build.sh')], cwd=ROOT)
        binary = ROOT / 'build/muzero_selfplay'
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f'Missing executable: {binary}')
    return binary


def model_path(data, iteration):
    return data / 'models' / f'model_{iteration + 1:08d}.pt'


def ensure_export(data, config, iteration):
    import torch
    from .train import create_model, export_model
    destination = model_path(data, iteration)
    if not destination.exists():
        checkpoint = torch.load(data / 'checkpoints/latest.pt', map_location='cpu', weights_only=True)
        if checkpoint['protocol_version'] != VERSION or checkpoint['iteration'] != iteration or checkpoint['identity'] != model_identity(config):
            raise ValueError('Cannot export a different checkpoint generation')
        model = create_model(config)
        model.load_state_dict(checkpoint['model'])
        export_model(model, destination)
    with atomic_path(data / 'models/latest.pt') as temporary:
        shutil.copyfile(destination, temporary)
    return destination


def initialize(data, config):
    import torch
    from .train import create_model, create_optimizer, save_checkpoint, seed_all
    checkpoint_path = data / 'checkpoints/latest.pt'
    if checkpoint_path.exists():
        return
    if any((data / name).exists() and any((data / name).glob('**/*.mzg' if name == 'selfplay' else '*.pt'))
           for name in ('models', 'selfplay')):
        raise ValueError('Checkpoint is missing from an existing lineage')
    seed_all(config['SEED'])
    model = create_model(config)
    if config['INIT_MODEL']:
        initial = torch.load(config['INIT_MODEL'], map_location='cpu', weights_only=True)
        if initial['protocol_version'] != VERSION or initial['identity'] != model_identity(config):
            raise ValueError('Shared initialization architecture mismatch')
        model.load_state_dict(initial['model'])
    optimizer = create_optimizer(model, config)
    save_checkpoint(checkpoint_path, model, optimizer, config, -1, {})


def train_and_checkpoint(data, c, iteration, records):
    import torch
    from .replay import Replay, game_statistics
    from .train import create_model, create_optimizer, load_checkpoint, save_checkpoint, train_iteration
    checkpoint_path = data / 'checkpoints/latest.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['protocol_version'] != VERSION or checkpoint['identity'] != model_identity(c):
        raise ValueError('Checkpoint model/data protocol mismatch')
    if checkpoint['iteration'] == iteration:
        return checkpoint['metrics']
    if checkpoint['iteration'] != iteration - 1:
        raise ValueError('Checkpoint is inconsistent with the pending iteration')
    del checkpoint
    started = time.monotonic()
    total_rows = sum(record.rows for record in records)
    ready = total_rows >= max(c['MIN_ROWS'], c['BATCH_SIZE'])
    device = c['DEVICE'] if ready else 'cpu'
    model = create_model(c).to(device)
    optimizer = create_optimizer(model, c)
    load_checkpoint(checkpoint_path, model, optimizer, c, device)
    metrics = {'steps': 0, 'loss': None, 'policy_loss': None, 'value_loss': None, 'replay_rows': 0}
    if ready:
        replay = Replay(records, c)
        metrics.update(train_iteration(model, optimizer, replay, c, iteration, device))
        metrics['replay_rows'] = replay.rows
        metrics['replay_groups'] = game_statistics(replay.records, replay.games)
    metrics['train_seconds'] = time.monotonic() - started
    save_checkpoint(checkpoint_path, model, optimizer, c, iteration, metrics)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def plan_games(state, config, total_rows):
    if config['SELFPLAY_SCHEDULE'] == 'fixed':
        return config['GAMES_PER_ITER']
    if state['completed_games'] == 0:
        return config['BOOTSTRAP_GAMES']
    if not state['target_initialized']:
        state['target_rows'] = max(config['MIN_ROWS'], config['BATCH_SIZE'])
        state['target_initialized'] = True
        factor = config['BACKFILL_FACTOR']
    else:
        state['target_rows'] += config['TRAIN_STEPS'] * config['BATCH_SIZE'] / config['REPLAY_RATIO']
        factor = 1.0
    deficit = max(0, state['target_rows'] - total_rows)
    return math.ceil(deficit * factor / state['rows_per_game'])


def run(config, binary, limit=None):
    import torch
    from .replay import GameRecord, catalog, game_statistics
    data = Path(config['DATA_DIR'])
    with run_lock(data / 'logs/run.lock'):
        immutable = {key: value for key, value in config.items() if key not in MUTABLE_KEYS}
        manifest_path = data / 'logs/manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['protocol_version'] != VERSION or manifest['config'] != immutable:
                raise ValueError('DATA_DIR belongs to a different configuration; use a new DATA_DIR')
        else:
            if any((data / name).exists() and any((data / name).iterdir()) for name in ('models', 'checkpoints', 'selfplay')):
                raise ValueError('Refusing to adopt artifacts without a manifest')
            write_json(manifest_path, {'protocol_version': VERSION, 'config': immutable})
        torch.set_num_threads(config['TORCH_THREADS'])
        state_path = data / 'logs/state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'iteration': 0, 'elapsed_seconds': 0, 'pending': None, 'target_rows': 0,
            'target_initialized': False, 'completed_games': 0,
            'rows_per_game': sum(size * size * weight for size, weight in
                                 zip(config['BOARD_SIZES'], config['BOARD_SIZE_WEIGHTS'])) / sum(config['BOARD_SIZE_WEIGHTS']),
        }
        write_json(state_path, state)
        initialize(data, config)
        resolved = data / 'logs/resolved.cfg'
        with atomic_path(resolved) as temporary:
            temporary.write_text(render_native_config(config))
        base_elapsed, started = state['elapsed_seconds'], time.monotonic()
        limit = limit if limit is not None else config['MAX_ITERS'] or None
        try:
            while True:
                state['elapsed_seconds'] = base_elapsed + time.monotonic() - started
                if state['pending'] is None and ((limit is not None and state['iteration'] >= limit) or
                        (config['MAX_TIME_SECONDS'] and state['elapsed_seconds'] >= config['MAX_TIME_SECONDS'])):
                    break
                iteration = state['iteration']
                directory = data / 'selfplay' / f'iter_{iteration:08d}'
                directory.mkdir(parents=True, exist_ok=True)
                if state['pending'] is None:
                    records = catalog(data, config['CANVAS_SIZE'])
                    if any(directory.glob('game_*.mzg')):
                        raise ValueError('Selfplay artifacts exist without an iteration plan')
                    planned = dict(state)
                    games = plan_games(planned, config, sum(record.rows for record in records))
                    planned['pending'] = {'games': games, 'seed': config['SEED'] + iteration * 1000000007,
                                          'selfplay_seconds': 0, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
                    state = planned
                    write_json(state_path, state)
                pending = state['pending']
                if pending['binary_sha256'] != hashlib.sha256(binary.read_bytes()).hexdigest():
                    raise ValueError('Selfplay binary changed during an unfinished iteration')
                checkpoint = torch.load(data / 'checkpoints/latest.pt', map_location='cpu', weights_only=True)
                generation = checkpoint['iteration']
                del checkpoint
                if generation not in (iteration - 1, iteration):
                    raise ValueError('Invalid checkpoint generation')
                expected = {directory / f'game_{index:08d}.mzg' for index in range(pending['games'])}
                existing = set(directory.glob('game_*.mzg'))
                if existing - expected:
                    raise ValueError('Unexpected selfplay game IDs')
                for path in existing:
                    GameRecord.inspect(path, config['CANVAS_SIZE']).load()
                if existing != expected:
                    if generation != iteration - 1:
                        raise ValueError('Trained checkpoint exists before selfplay completion')
                    deployed = ensure_export(data, config, generation)
                    print(f"iter={iteration} selfplay={len(existing)}/{pending['games']}", flush=True)
                    selfplay_start = time.monotonic()
                    try:
                        run_process([str(binary), str(resolved), str(deployed), str(directory),
                                     str(pending['games']), str(pending['seed']), '0'])
                    finally:
                        pending['selfplay_seconds'] += time.monotonic() - selfplay_start
                        state['elapsed_seconds'] = base_elapsed + time.monotonic() - started
                        write_json(state_path, state)
                    if set(directory.glob('game_*.mzg')) != expected:
                        raise ValueError('Selfplay did not produce the planned games')
                records = catalog(data, config['CANVAS_SIZE'])
                current = [record for record in records if record.path.parent == directory]
                current_groups = game_statistics(current, (record.load() for record in current))
                metrics = train_and_checkpoint(data, config, iteration, records)
                ensure_export(data, config, iteration)
                rows = sum(record.rows for record in current)
                metrics = dict(metrics, game_groups=current_groups, iteration=iteration, games=len(current), rows=rows,
                               total_rows=sum(record.rows for record in records),
                               black_wins=sum(record.winner == 1 for record in current),
                               draws=sum(record.winner == 0 for record in current),
                               white_wins=sum(record.winner == -1 for record in current),
                               selfplay_seconds=pending['selfplay_seconds'])
                write_json(data / 'logs/iters' / f'{iteration:08d}.json', metrics)
                state = dict(state, rows_per_game=rows / len(current) if current else state['rows_per_game'],
                             completed_games=state['completed_games'] + len(current), pending=None,
                             iteration=iteration + 1, elapsed_seconds=base_elapsed + time.monotonic() - started)
                write_json(state_path, state)
                print(json.dumps(metrics, sort_keys=True), flush=True)
        finally:
            state['elapsed_seconds'] = base_elapsed + time.monotonic() - started
            write_json(state_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('max_iters', type=int, nargs='?')
    parser.add_argument('--config-dir', default=os.environ.get('CONFIG_DIR', 'configs/baseline'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.max_iters is not None and args.max_iters < 0:
        parser.error('max_iters must be nonnegative')
    config = load_config(args.config_dir)
    if args.dry_run:
        print(render_config(config), end='')
        return
    install_signals()
    binary = build_binary()
    run(config, binary, args.max_iters)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
