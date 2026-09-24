import argparse
import configparser
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILE_SECTIONS = {
    'env.cfg': ('env',),
    'selfplay.cfg': ('selfplay', 'search', 'noise', 'fpu', 'lcb', 'temperature'),
    'net.cfg': ('model',),
    'train.cfg': ('parallel', 'training', 'optimizer', 'loss', 'replay'),
    'run.cfg': ('run',),
}
CONFIG_FILES = tuple(FILE_SECTIONS)
GROUPS = {
    'run': 'SEED DEVICE MAX_ITERS MAX_TIME_SECONDS DATA_DIR INIT_MODEL'.split(),
    'env': 'BOARD_SIZE'.split(),
    'model': 'NUM_BLOCKS NUM_CHANNELS VALUE_HEAD AUXILIARY_POLICY_HEADS'.split(),
    'parallel': 'TORCH_THREADS NUM_GAME_THREADS NN_MAX_BATCH_SIZE NN_BATCH_WAIT_US'.split(),
    'selfplay': 'SELFPLAY_SCHEDULE GAMES_PER_ITER BOOTSTRAP_GAMES BACKFILL_FACTOR'.split(),
    'search': 'FULL_SEARCH_VISITS CHEAP_SEARCH_VISITS CHEAP_SEARCH_PROB PB_C_INIT PB_C_BASE'.split(),
    'noise': 'DIRICHLET_TOTAL_CONCENTRATION DIRICHLET_NOISE_WEIGHT SHAPED_DIRICHLET_NOISE'.split(),
    'fpu': 'USE_FPU FPU_REDUCTION_MAX ROOT_FPU_REDUCTION_MAX FPU_PARENT_WEIGHT_BY_VISITED_POLICY_POW'.split(),
    'lcb': 'USE_LCB_FOR_SELECTION LCB_STDEVS MIN_VISIT_PROP_FOR_LCB'.split(),
    'temperature': 'MOVE_TEMPERATURE_SCHEDULE TEMPERATURE_MOVES CHOSEN_MOVE_TEMPERATURE_EARLY CHOSEN_MOVE_TEMPERATURE CHOSEN_MOVE_TEMPERATURE_HALFLIFE ROOT_POLICY_TEMPERATURE_EARLY ROOT_POLICY_TEMPERATURE'.split(),
    'training': 'BATCH_SIZE TRAIN_STEPS UNROLL_STEPS HIDDEN_GRADIENT_SCALE SYMMETRY_AUGMENTATION'.split(),
    'optimizer': 'LR WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 ADAM_EPS'.split(),
    'loss': 'VALUE_LOSS_SCALE SOFT_POLICY_LOSS_SCALE OPPONENT_POLICY_LOSS_SCALE SOFT_POLICY_TEMPERATURE SOFT_POLICY_EPS'.split(),
    'replay': 'REPLAY_RATIO REPLAY_WINDOW FIXED_WINDOW_ROWS MIN_ROWS MAX_ROWS TAPER_WINDOW_EXPONENT EXPAND_WINDOW_PER_ROW'.split(),
}
INT_KEYS = set('BOARD_SIZE NUM_BLOCKS NUM_CHANNELS SEED TORCH_THREADS NUM_GAME_THREADS NN_MAX_BATCH_SIZE NN_BATCH_WAIT_US FULL_SEARCH_VISITS CHEAP_SEARCH_VISITS BATCH_SIZE TRAIN_STEPS UNROLL_STEPS MIN_ROWS MAX_ROWS BOOTSTRAP_GAMES MAX_ITERS MAX_TIME_SECONDS TEMPERATURE_MOVES FIXED_WINDOW_ROWS GAMES_PER_ITER'.split())
BOOL_KEYS = set('SHAPED_DIRICHLET_NOISE USE_LCB_FOR_SELECTION SYMMETRY_AUGMENTATION AUXILIARY_POLICY_HEADS USE_FPU'.split())
STR_KEYS = set('DEVICE DATA_DIR INIT_MODEL VALUE_HEAD MOVE_TEMPERATURE_SCHEDULE REPLAY_WINDOW SELFPLAY_SCHEDULE'.split())
KEYS = {key for group in GROUPS.values() for key in group}
FLOAT_KEYS = KEYS - INT_KEYS - BOOL_KEYS - STR_KEYS
EXP_KEYS = {'MAX_ITERS', 'MAX_TIME_SECONDS', 'SHARED_INIT', 'ARM_GPUS'}


def boolean(value):
    if value.lower() in {'true', '1'}:
        return True
    if value.lower() in {'false', '0'}:
        return False
    raise ValueError(f'Expected true/false, got {value!r}')


def read_profile(path, experiment=False):
    filename = Path(path).name.removesuffix('.local')
    allowed_sections = ('experiment',) if filename == 'exp.cfg' and experiment else FILE_SECTIONS.get(filename)
    if allowed_sections is None:
        raise ValueError(f'{path}: unsupported configuration filename')
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        parser.read_string('[profile]\n' + Path(path).read_text(), source=str(path))
    except configparser.Error as error:
        raise ValueError(str(error)) from error
    if parser.defaults() or set(parser['profile']) - {'extends'}:
        raise ValueError(f'{path}: only extends may appear before the first section')
    parent = parser['profile'].get('extends')
    if filename == 'exp.cfg' and parent:
        raise ValueError(f'{path}: experiment umbrellas do not support extends')
    groups = GROUPS | {'experiment': EXP_KEYS}
    values = {}
    for section in parser.sections():
        if section == 'profile':
            continue
        if section not in allowed_sections:
            raise ValueError(f'{path}: section [{section}] does not belong in {filename}')
        for key, value in parser[section].items():
            internal = key.upper()
            if key != key.lower() or internal not in groups[section]:
                raise ValueError(f'{path}: unknown or duplicate key [{section}] {key}')
            if '\n' in value or '$' in value or '`' in value:
                raise ValueError(f'{path}: values must be single-line literals')
            if (section == 'experiment') == experiment:
                values[internal] = value
    return parent, values


def config_chain(directory, seen=None):
    directory = Path(directory).resolve()
    seen = set() if seen is None else seen
    if directory in seen:
        raise ValueError(f'Configuration inheritance cycle: {directory}')
    seen.add(directory)
    parent, _ = read_profile(directory / 'run.cfg')
    chain = []
    if parent:
        parent = Path(parent)
        chain = config_chain(parent if parent.is_absolute() else ROOT / 'configs' / parent, seen)
    return chain + [directory]


def validate(c):
    nonnegative = set('TEMPERATURE_MOVES SEED NUM_BLOCKS NN_BATCH_WAIT_US UNROLL_STEPS MAX_ROWS MAX_ITERS MAX_TIME_SECONDS'.split())
    for key in INT_KEYS:
        if c[key] < (0 if key in nonnegative else 1):
            raise ValueError(f'{key} out of range')
    zero_allowed = set('CHEAP_SEARCH_PROB PB_C_INIT DIRICHLET_NOISE_WEIGHT MIN_VISIT_PROP_FOR_LCB CHOSEN_MOVE_TEMPERATURE_EARLY CHOSEN_MOVE_TEMPERATURE FPU_REDUCTION_MAX ROOT_FPU_REDUCTION_MAX WEIGHT_DECAY VALUE_LOSS_SCALE SOFT_POLICY_LOSS_SCALE OPPONENT_POLICY_LOSS_SCALE HIDDEN_GRADIENT_SCALE ADAM_BETA1 ADAM_BETA2'.split())
    for key in FLOAT_KEYS:
        value = c[key]
        if not math.isfinite(value) or value < 0 or (value == 0 and key not in zero_allowed):
            raise ValueError(f'{key} out of range')
    for key in 'CHEAP_SEARCH_PROB DIRICHLET_NOISE_WEIGHT MIN_VISIT_PROP_FOR_LCB HIDDEN_GRADIENT_SCALE'.split():
        if c[key] > 1:
            raise ValueError(f'{key} must be <= 1')
    if c['ADAM_BETA1'] >= 1 or c['ADAM_BETA2'] >= 1:
        raise ValueError('Adam betas must be < 1')
    if c['BOARD_SIZE'] < 5:
        raise ValueError('Gomoku requires BOARD_SIZE>=5')
    if c['BOARD_SIZE'] > 25 or c['NUM_CHANNELS'] < 2:
        raise ValueError('BOARD_SIZE must be <=25 and NUM_CHANNELS >=2')
    if c['MAX_ROWS'] and c['MAX_ROWS'] < max(c['MIN_ROWS'], c['BATCH_SIZE']):
        raise ValueError('MAX_ROWS must be 0 or >= MIN_ROWS and BATCH_SIZE')
    if c['BACKFILL_FACTOR'] < 1:
        raise ValueError('BACKFILL_FACTOR must be >=1')
    if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', c['DEVICE']):
        raise ValueError('DEVICE must be cpu or cuda[:index]')
    for key, choices in {
        'VALUE_HEAD': {'scalar', 'wdl'},
        'MOVE_TEMPERATURE_SCHEDULE': {'exponential', 'threshold'},
        'REPLAY_WINDOW': {'power_law', 'fixed'},
        'SELFPLAY_SCHEDULE': {'adaptive', 'fixed'},
    }.items():
        if c[key] not in choices:
            raise ValueError(f'{key} must be one of {sorted(choices)}')
    if not c['AUXILIARY_POLICY_HEADS'] and (c['SOFT_POLICY_LOSS_SCALE'] or c['OPPONENT_POLICY_LOSS_SCALE']):
        raise ValueError('Disabled auxiliary heads require zero auxiliary loss scales')
    if c['REPLAY_WINDOW'] == 'fixed' and c['FIXED_WINDOW_ROWS'] < max(c['BATCH_SIZE'], c['MIN_ROWS']):
        raise ValueError('FIXED_WINDOW_ROWS must be >= MIN_ROWS and BATCH_SIZE')


def load_config(directory, environ=None):
    directory = Path(directory)
    if not directory.is_absolute():
        directory = ROOT / directory
    values = {}
    layers = [(layer, '') for layer in config_chain(directory)] + [(directory, '.local')]
    for layer, suffix in layers:
        merged = {}
        for filename in CONFIG_FILES:
            path = layer / (filename + suffix)
            if not path.exists():
                continue
            parent, entries = read_profile(path)
            if parent and (filename != 'run.cfg' or suffix):
                raise ValueError(f'{path}: extends belongs in run.cfg')
            overlap = merged.keys() & entries.keys()
            if overlap:
                raise ValueError(f'{layer}: duplicate keys across files: {sorted(overlap)}')
            merged.update(entries)
        values.update(merged)
    missing = KEYS - values.keys()
    if missing:
        raise ValueError(f'Missing keys: {sorted(missing)}')
    env = os.environ if environ is None else environ
    values.update({key: env[key] for key in KEYS & env.keys()})
    typed = {}
    for key, value in values.items():
        if key in BOOL_KEYS:
            typed[key] = boolean(value)
        elif key in INT_KEYS:
            typed[key] = int(value)
        elif key in FLOAT_KEYS:
            typed[key] = float(value)
        else:
            typed[key] = value
    validate(typed)
    if typed['DATA_DIR'] == 'auto':
        try:
            suffix = directory.resolve().relative_to(ROOT / 'configs')
        except ValueError:
            suffix = directory.resolve().name
        typed['DATA_DIR'] = str(ROOT / 'data' / suffix)
    else:
        path = Path(typed['DATA_DIR'])
        typed['DATA_DIR'] = str((path if path.is_absolute() else ROOT / path).resolve())
    if typed['INIT_MODEL']:
        path = Path(typed['INIT_MODEL'])
        typed['INIT_MODEL'] = str((path if path.is_absolute() else ROOT / path).resolve())
    return typed


def render_config(config, filename=None):
    lines = []
    sections = FILE_SECTIONS[filename] if filename else GROUPS
    for section in sections:
        keys = GROUPS[section]
        lines.append(f'[{section}]')
        for key in keys:
            value = str(config[key]).lower() if key in BOOL_KEYS else str(config[key])
            lines.append(f'{key.lower()} = {value}'.rstrip())
        lines.append('')
    return '\n'.join(lines)


def render_native_config(config):
    return ''.join(f'{key}={int(value) if key in BOOL_KEYS else value}\n' for key, value in sorted(config.items()))


def model_identity(c):
    return {key: c[key] for key in ('BOARD_SIZE', 'NUM_BLOCKS', 'NUM_CHANNELS', 'VALUE_HEAD', 'AUXILIARY_POLICY_HEADS')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', default='configs/baseline')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = load_config(args.config_dir, environ={})
    text = render_native_config(config) if args.native else render_config(config)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
