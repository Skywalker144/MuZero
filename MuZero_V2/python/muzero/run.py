import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .protocol import VERSION
from .config import CONSISTENCY_KEYS, ROOT, load_config, model_identity, render_config, render_native_config
from .process import install_signals, run_process
from .storage import IterationStore, RunState, atomic_path, checkpoint_path, model_path, run_lock, write_json

MUTABLE_KEYS = set('DEVICE TORCH_THREADS NUM_GAME_THREADS NN_MAX_BATCH_SIZE NN_BATCH_WAIT_US MAX_ITERS MAX_TIME_SECONDS DATA_DIR INIT_MODEL'.split())
MUTABLE_KEYS |= CONSISTENCY_KEYS


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


def ensure_export(data, config, iteration):
    import torch
    from .train import create_model, export_model, load_model_state
    destination = model_path(data, iteration)
    if not destination.exists():
        checkpoint = torch.load(checkpoint_path(data, iteration), map_location='cpu', weights_only=True)
        if checkpoint['protocol_version'] != VERSION or checkpoint['iteration'] != iteration or checkpoint['identity'] != model_identity(config):
            raise ValueError('Cannot export a different checkpoint generation')
        model = create_model(config)
        load_model_state(model, checkpoint['average']['model'])
        export_model(model, destination)
    return destination


def initialize(data, config):
    import torch
    from .train import (create_model, create_optimizer, load_model_state, load_optimizer_state,
                        save_checkpoint, seed_all, training_steps, WeightAverage)
    destination = checkpoint_path(data, -1)
    if destination.exists():
        return
    if any((data / name).exists() and any((data / name).glob('**/*.mzs' if name == 'selfplay' else '*.pt'))
           for name in ('models', 'selfplay')):
        raise ValueError('Checkpoint is missing from an existing lineage')
    seed_all(config['SEED'])
    model = create_model(config)
    steps = 0
    if config['INIT_MODEL']:
        initial = torch.load(config['INIT_MODEL'], map_location='cpu', weights_only=True)
        if initial['protocol_version'] != VERSION or initial['identity'] != model_identity(config):
            raise ValueError('Shared initialization architecture mismatch')
        load_model_state(model, initial['model'])
        steps = training_steps(initial)
    optimizer = create_optimizer(model, config)
    average = WeightAverage(model, config)
    if config['INIT_MODEL']:
        if 'optimizer' in initial:
            load_optimizer_state(optimizer, initial['optimizer'])
        average.load_state_dict(initial['average'])
    save_checkpoint(destination, model, optimizer, config, -1, {}, steps, average)


def train_and_checkpoint(data, c, iteration, records):
    import torch
    from .replay import Replay, game_statistics
    from .train import create_model, create_optimizer, load_checkpoint, save_checkpoint, seed_all, train_iteration, training_steps, WeightAverage
    source = checkpoint_path(data, iteration - 1)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if checkpoint['protocol_version'] != VERSION or checkpoint['identity'] != model_identity(c):
        raise ValueError('Checkpoint model/data protocol mismatch')
    steps = training_steps(checkpoint)
    if checkpoint['iteration'] != iteration - 1:
        raise ValueError('Checkpoint is inconsistent with the committed iteration')
    del checkpoint
    started = time.monotonic()
    total_rows = sum(record.rows for record in records)
    ready = total_rows >= max(c['MIN_ROWS'], c['BATCH_SIZE'])
    device = c['DEVICE'] if ready else 'cpu'
    seed_all(c['SEED'] + iteration + 1)
    model = create_model(c).to(device)
    optimizer = create_optimizer(model, c)
    average = WeightAverage(model, c)
    load_checkpoint(source, model, optimizer, c, average)
    metrics = {'steps': 0, 'loss': None, 'policy_loss': None, 'value_loss': None, 'consistency_loss': None,
               'use_consistency_loss': c['USE_CONSISTENCY_LOSS'],
               'consistency_loss_scale': c['CONSISTENCY_LOSS_SCALE'], 'replay_rows': 0}
    if ready:
        replay_started = time.monotonic()
        replay = Replay(records, c, data / 'replay' / f'{iteration:08d}.json')
        metrics['replay_load_seconds'] = time.monotonic() - replay_started
        if replay.weight_sum > 0:
            metrics.update(train_iteration(model, optimizer, replay, c, iteration, device, average))
        metrics['replay_rows'] = replay.rows
        metrics['replay_groups'] = game_statistics(replay.records, replay.games)
    metrics['train_seconds'] = time.monotonic() - started
    save_checkpoint(checkpoint_path(data, iteration), model, optimizer, c, iteration, metrics, steps + metrics['steps'], average)
    del model, optimizer, average
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def plan_games(state, config, total_rows):
    if config['SELFPLAY_SCHEDULE'] == 'fixed':
        return config['GAMES_PER_ITER']
    if state['iteration'] == 0 or state['rows_per_game'] == 0:
        return config['BOOTSTRAP_GAMES']
    if state['iteration'] == 1:
        deficit = max(0, max(config['MIN_ROWS'], config['BATCH_SIZE']) - total_rows)
        return math.ceil(deficit * config['BACKFILL_FACTOR'] / state['rows_per_game'])
    if state['iteration'] == 2:
        state['target_rows'] = total_rows
    state['target_rows'] += config['TRAIN_STEPS'] * config['BATCH_SIZE'] / config['REPLAY_RATIO']
    return max(1, math.ceil((state['target_rows'] - total_rows) / state['rows_per_game']))


def run(config, binary, limit=None):
    import torch
    from .plots import render_training
    from .replay import catalog, directory_records, game_statistics
    from .train import training_steps
    data = Path(config['DATA_DIR'])
    with run_lock(data / 'logs/run.lock'):
        immutable = {key: value for key, value in config.items() if key not in MUTABLE_KEYS}
        manifest_path = data / 'logs/manifest.json'
        manifest = {'protocol_version': VERSION, 'iteration_commit_version': 1, 'config': immutable}
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('DATA_DIR belongs to a different configuration or commit format; use a new DATA_DIR')
        else:
            if any((data / name).exists() and any((data / name).iterdir()) for name in ('models', 'checkpoints', 'selfplay')):
                raise ValueError('Refusing to adopt artifacts without a manifest')
            write_json(manifest_path, manifest)
        torch.set_num_threads(config['TORCH_THREADS'])
        store = IterationStore(data)
        state = store.read()
        if state is None:
            initialize(data, config)
            state = RunState(
                iteration=0, elapsed_seconds=0.0, target_rows=0, completed_games=0,
                rows_per_game=sum(size * size * weight for size, weight in
                                  zip(config['BOARD_SIZES'], config['BOARD_SIZE_WEIGHTS'])) / sum(config['BOARD_SIZE_WEIGHTS']),
            )
            store.commit(state)
        store.recover(state)
        resolved = data / 'logs/resolved.cfg'
        with atomic_path(resolved) as temporary:
            temporary.write_text(render_native_config(config))
        if (data / 'logs/iters').is_dir():
            render_training(data)
        limit = limit if limit is not None else config['MAX_ITERS'] or None
        records = None
        while ((limit is None or state['iteration'] < limit) and
               (not config['MAX_TIME_SECONDS'] or state['elapsed_seconds'] < config['MAX_TIME_SECONDS'])):
            started = time.monotonic()
            if records is None:
                records = catalog(data, config['CANVAS_SIZE'])
            iteration = state['iteration']
            checkpoint = torch.load(checkpoint_path(data, iteration - 1), map_location='cpu', weights_only=True)
            if checkpoint['protocol_version'] != VERSION or checkpoint['identity'] != model_identity(config):
                raise ValueError('Checkpoint model/data protocol mismatch')
            if checkpoint['iteration'] != iteration - 1:
                raise ValueError('Invalid committed checkpoint generation')
            steps = training_steps(checkpoint)
            del checkpoint
            planned: RunState = state.copy()
            games = plan_games(planned, config, sum(record.rows for record in records))
            evaluator = config['BOOTSTRAP_EVALUATOR'] if steps == 0 else 'network'
            seed = config['SEED'] + iteration * 1000000007
            directory = data / 'selfplay' / f'iter_{iteration:08d}'
            write_json(directory / 'manifest.json', {
                'games': games, 'seed': seed, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'evaluator': evaluator, 'model_generation': iteration - 1,
            })
            selfplay_seconds = 0.0
            if games:
                deployed = ensure_export(data, config, iteration - 1) if evaluator == 'network' else '-'
                print(f"iter={iteration} evaluator={evaluator} selfplay=0/{games}", flush=True)
                selfplay_start = time.monotonic()
                run_process([str(binary), str(resolved), str(deployed), str(directory),
                             str(games), str(seed), '0', evaluator])
                selfplay_seconds = time.monotonic() - selfplay_start
            current = directory_records(directory, config['CANVAS_SIZE'])
            if {record.game_id for record in current} != set(range(games)):
                raise ValueError('Selfplay did not produce the planned games')
            records = records + current
            current_groups = game_statistics(current, (record.load() for record in current))
            metrics = train_and_checkpoint(data, config, iteration, records)
            if steps > 0 or metrics['steps'] > 0 or config['BOOTSTRAP_EVALUATOR'] == 'network':
                ensure_export(data, config, iteration)
            rows = sum(record.rows for record in current)
            metrics = dict(metrics, game_groups=current_groups, iteration=iteration, games=len(current), rows=rows,
                           selfplay_evaluator=evaluator, model_generation=iteration - 1,
                           total_rows=sum(record.rows for record in records),
                           black_wins=sum(record.winner == 1 for record in current),
                           draws=sum(record.winner == 0 for record in current),
                           white_wins=sum(record.winner == -1 for record in current),
                           selfplay_seconds=selfplay_seconds)
            metric_path = data / 'logs/iters' / f'{iteration:08d}.json'
            write_json(metric_path, metrics)
            render_training(data)
            elapsed = state['elapsed_seconds'] + time.monotonic() - started
            metrics['elapsed_seconds'] = elapsed
            write_json(metric_path, metrics)
            planned.update(rows_per_game=rows / len(current) if current else state['rows_per_game'],
                           completed_games=state['completed_games'] + len(current),
                           iteration=iteration + 1, elapsed_seconds=elapsed)
            store.commit(planned)
            state = planned
            print(json.dumps(metrics, sort_keys=True), flush=True)


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
