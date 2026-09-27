from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Condition
import time
from typing import Any
from uuid import uuid4

from muzero.engine import Engine
from muzero.protocol import SPEC


class Conflict(ValueError):
    pass


def integer(payload: dict, name: str, minimum: int, maximum: int) -> int:
    value = payload.get(name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} 必须是 {minimum}–{maximum} 之间的整数')
    return value


class App:
    def __init__(self, binary: Path, models: dict[str, Path], config: dict[str, Any], default_size: int):
        self.binary, self.models, self.config = binary, models, config
        self.default_size = default_size
        self.engine: Engine | None = None
        self.condition = Condition()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='muzero-game')
        self.closed = False
        self.state = dict(instance=uuid4().hex, version=0, busy=False, phase='idle', game=None, analysis=None,
                          error=None, model=None, human=1, rule='freestyle',
                          visits=config['FULL_SEARCH_VISITS'], started_at=None)

    def catalog(self) -> dict:
        return dict(models=[dict(id=key, label=key) for key in self.models], rules=SPEC['rules'],
                    default_size=self.default_size, default_visits=self.config['FULL_SEARCH_VISITS'],
                    search_threads=self.config['NUM_SEARCH_THREADS'], virtual_loss=self.config['VIRTUAL_LOSS'],
                    device=self.config['DEVICE'])

    def snapshot(self, since: int = -1) -> dict:
        with self.condition:
            if since >= 0:
                self.condition.wait_for(lambda: self.closed or self.state['version'] != since, timeout=20)
            return deepcopy(self.state)

    def publish(self, **values):
        with self.condition:
            self.state.update(values)
            self.state['version'] += 1
            self.condition.notify_all()

    def submit(self, operation: str, payload: dict) -> dict:
        with self.condition:
            if type(payload.get('version')) is not int:
                raise ValueError('缺少有效的棋局版本号')
            if self.closed or self.state['busy'] or payload['version'] != self.state['version']:
                raise Conflict('棋局状态已更新，请等待同步后再操作')
            game = self.state['game']
            if operation == 'new':
                if payload.get('model') not in self.models:
                    raise ValueError('请选择可用模型')
                integer(payload, 'size', 5, 25)
                integer(payload, 'visits', 1, 100000)
                if type(payload.get('human')) is not int or payload['human'] not in (-1, 1):
                    raise ValueError('执子必须为黑或白')
                if payload.get('rule') not in SPEC['rules']:
                    raise ValueError('未知棋规')
            elif operation == 'play':
                if not game or game['finished'] or game['player'] != self.state['human']:
                    raise Conflict('当前不能落子')
                action = integer(payload, 'action', 0, game['board_size'] ** 2 - 1)
                if game['board'][action]:
                    raise ValueError('此处已有棋子')
            elif operation == 'undo':
                if not game or not any((1 if i % 2 == 0 else -1) == self.state['human'] for i in range(len(game['moves']))):
                    raise ValueError('还没有可以撤回的人类落子')
            elif operation == 'retry':
                if not game or game['finished'] or game['player'] == self.state['human']:
                    raise Conflict('当前不需要 AI 落子')
            else:
                raise ValueError('未知操作')
            self.publish(busy=True, phase='loading' if operation == 'new' else 'thinking',
                         error=None, started_at=time.time())
            self.executor.submit(self.perform, operation, dict(payload))
            return deepcopy(self.state)

    def perform(self, operation: str, payload: dict):
        try:
            if operation == 'new':
                selected = payload['model']
                replacement = self.engine is None or self.engine.process.poll() is not None or selected != self.state['model']
                engine = Engine(self.binary, self.models[selected], self.config) if replacement else self.engine
                try:
                    if payload['size'] > engine.canvas:
                        raise ValueError(f'该模型最大支持 {engine.canvas}×{engine.canvas} 棋盘')
                    reply = engine.command(f"new {payload['size']} {payload['rule']}")
                except BaseException:
                    if replacement:
                        engine.close()
                    raise
                previous, self.engine = self.engine, engine
                if replacement and previous:
                    previous.close()
                self.publish(game=reply['state'], analysis=None, model=selected,
                             human=payload['human'], rule=payload['rule'], visits=payload['visits'])
            elif operation == 'play':
                reply = self.engine.command(f"play {payload['action']}")
                self.publish(game=reply['state'], analysis=None)
            elif operation == 'undo':
                moves = self.state['game']['moves']
                last_human = max(i for i in range(len(moves)) if (1 if i % 2 == 0 else -1) == self.state['human'])
                reply = self.engine.command(f'undo {len(moves) - last_human}')
                self.publish(game=reply['state'], analysis=None)
            game = self.state['game']
            if game and not game['finished'] and game['player'] != self.state['human']:
                self.publish(phase='thinking', started_at=time.time())
                reply = self.engine.command(f"genmove {self.state['visits']}")
                self.publish(game=reply['state'], analysis=reply['analysis'])
            self.publish(busy=False, phase='idle', started_at=None)
        except Exception as error:
            self.publish(busy=False, phase='error', error=str(error), started_at=None)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.executor.shutdown(wait=True)
        if self.engine:
            self.engine.close()
