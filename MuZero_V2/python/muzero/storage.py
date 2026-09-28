import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TypedDict


class RunState(TypedDict):
    iteration: int
    elapsed_seconds: float
    target_rows: float
    completed_games: int
    rows_per_game: float


def checkpoint_path(data: Path, iteration: int) -> Path:
    return data / 'checkpoints' / f'checkpoint_{iteration + 1:08d}.pt'


def model_path(data: Path, iteration: int) -> Path:
    return data / 'models' / f'model_{iteration + 1:08d}.pt'


class IterationStore:
    def __init__(self, data: Path):
        self.data = data
        self.state_path = data / 'logs/state.json'

    def read(self) -> RunState | None:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    def publish(self, state: RunState) -> None:
        for folder, source in (
            ('checkpoints', checkpoint_path(self.data, state['iteration'] - 1)),
            ('models', model_path(self.data, state['iteration'] - 1)),
        ):
            destination = self.data / folder / 'latest.pt'
            if folder == 'models' and not source.exists():
                destination.unlink(missing_ok=True)
                continue
            with atomic_path(destination) as temporary:
                temporary.unlink()
                os.link(source, temporary)
        for path in (self.data / 'checkpoints').glob('checkpoint_*.pt'):
            if path != checkpoint_path(self.data, state['iteration'] - 1):
                path.unlink()

    def recover(self, state: RunState) -> None:
        committed = checkpoint_path(self.data, state['iteration'] - 1)
        if not committed.is_file():
            raise ValueError(f'Missing committed checkpoint: {committed}')
        for folder, pattern, prefix, offset in (
            ('selfplay', 'iter_*', 'iter_', 0),
            ('replay', '*.json', '', 0),
            ('logs/iters', '*.json', '', 0),
            ('models', 'model_*.pt', 'model_', 1),
        ):
            for path in (self.data / folder).glob(pattern):
                if int(path.stem.removeprefix(prefix)) >= state['iteration'] + offset:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
        for folder in ('selfplay', 'replay', 'logs', 'models', 'checkpoints'):
            for path in (self.data / folder).rglob('*.tmp'):
                path.unlink()
        self.publish(state)

    def commit(self, state: RunState) -> None:
        write_json(self.state_path, state)
        self.publish(state)


@contextlib.contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


@contextlib.contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Already running: {path.parent}') from error
        yield
