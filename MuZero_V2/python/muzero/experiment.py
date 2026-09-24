import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import CONFIG_FILES, EXP_KEYS, KEYS, ROOT, boolean, load_config, model_identity, read_profile, render_config
from .process import install_signals, stop_process
from .run import build_binary
from .storage import atomic_path, run_lock, write_json

def experiment_plan(directory, environ=None):
    env = dict(os.environ if environ is None else environ)
    directory = Path(directory)
    if not directory.is_absolute():
        directory = ROOT / directory
    _, settings = read_profile(directory / 'exp.cfg', experiment=True)
    if set(settings) != EXP_KEYS:
        raise ValueError(f'Missing experiment keys: {sorted(EXP_KEYS - set(settings))}')
    settings.update({key: env[key] for key in EXP_KEYS & env.keys()})
    settings['SHARED_INIT'] = boolean(settings['SHARED_INIT'])
    for key in ('MAX_ITERS', 'MAX_TIME_SECONDS'):
        settings[key] = int(settings[key])
    if settings['MAX_ITERS'] < 0 or settings['MAX_TIME_SECONDS'] < 0 or settings['SHARED_INIT'] not in (0, 1):
        raise ValueError('Invalid experiment limits or SHARED_INIT')
    arms = []
    for arm in sorted(directory.iterdir()):
        if not (arm / 'run.cfg').is_file():
            continue
        config = load_config(arm, environ=env)
        for key in ('MAX_ITERS', 'MAX_TIME_SECONDS'):
            config[key] = settings[key]
        if not config['MAX_ITERS'] and not config['MAX_TIME_SECONDS']:
            raise ValueError(f'Unbounded experiment: {arm}')
        arms.append({'name': arm.name, 'config': config})
    if not arms:
        raise ValueError(f'No experiment arms in {directory}')
    destinations = [Path(arm['config']['DATA_DIR']).resolve() for arm in arms]
    for i, destination in enumerate(destinations):
        for other in destinations[:i]:
            if destination == other or destination in other.parents or other in destination.parents:
                raise ValueError('Experiment arms must have independent DATA_DIR paths')
    if settings['ARM_GPUS']:
        slots = [gpu.strip() for gpu in settings['ARM_GPUS'].split(',')]
        if any(not gpu for gpu in slots):
            raise ValueError('ARM_GPUS contains an empty slot')
    else:
        slots = [None]
    return directory.resolve(), settings, arms, slots


def prepare_initializations(work, arms):
    import torch
    from .train import create_model, seed_all
    for arm in arms:
        config = arm['config']
        if config['INIT_MODEL']:
            raise ValueError('Use SHARED_INIT=0 when specifying INIT_MODEL manually')
        identity = model_identity(config)
        key = hashlib.sha256(json.dumps([identity, config['SEED']], sort_keys=True).encode()).hexdigest()
        destination = work / 'initializations' / f'{key}.pt'
        if not destination.exists():
            seed_all(config['SEED'])
            model = create_model(config)
            with atomic_path(destination) as temporary:
                torch.save({'identity': identity, 'model': model.state_dict()}, temporary)
        config['INIT_MODEL'] = str(destination)


def launch(work, arms, slots, binary):
    queue = list(arms)
    running = []
    try:
        while queue or running:
            occupied = {item['slot'] for item in running}
            for slot_index, gpu in enumerate(slots):
                if not queue or slot_index in occupied:
                    continue
                arm = queue.pop(0)
                config = arm['config']
                directory = work / 'configs' / arm['name']
                directory.mkdir(parents=True, exist_ok=True)
                for filename in CONFIG_FILES:
                    with atomic_path(directory / filename) as temporary:
                        temporary.write_text(render_config(config, filename))
                env = {key: value for key, value in os.environ.items() if key not in KEYS}
                env['CONFIG_DIR'] = str(directory)
                env['MUZERO_BINARY'] = str(binary)
                if gpu is not None:
                    env['CUDA_VISIBLE_DEVICES'] = gpu
                    env['DEVICE'] = 'cuda:0'
                output = (work / f"{arm['name']}.runner.log").open('a', buffering=1)
                try:
                    process = subprocess.Popen([sys.executable, '-m', 'muzero.run'], env=env,
                                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    output.close()
                    raise
                running.append({'process': process, 'output': output, 'slot': slot_index, 'name': arm['name']})
                print(f"started {arm['name']} gpu={gpu or config['DEVICE']}", flush=True)
            for item in list(running):
                code = item['process'].poll()
                if code is None:
                    continue
                item['output'].close()
                running.remove(item)
                if code:
                    raise RuntimeError(f"Experiment {item['name']} failed ({code}); see {work / (item['name'] + '.runner.log')}")
                print(f"completed {item['name']}", flush=True)
            if running:
                time.sleep(0.2)
    finally:
        for item in running:
            stop_process(item['process'])
            item['output'].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', default=os.environ.get('CONFIG_DIR'))
    parser.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN') == '1')
    args = parser.parse_args()
    if not args.config_dir:
        parser.error('Set CONFIG_DIR to an experiment umbrella')
    directory, settings, arms, slots = experiment_plan(args.config_dir)
    if args.dry_run:
        print(json.dumps({'umbrella': str(directory), 'settings': settings, 'slots': slots, 'arms': arms}, indent=2))
        return
    install_signals()
    tag = hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    work = ROOT / 'data/experiments' / f'{directory.name}_{tag}'
    with run_lock(work / 'experiment.lock'):
        binary = build_binary()
        if settings['SHARED_INIT']:
            prepare_initializations(work, arms)
        write_json(work / 'plan.json', {'umbrella': str(directory), 'settings': settings, 'arms': arms, 'slots': slots})
        launch(work, arms, slots, binary)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
