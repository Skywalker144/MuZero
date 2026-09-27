import json
from pathlib import Path
import select
import subprocess
import tempfile
from typing import Any

from .config import render_native_config


class Engine:
    def __init__(self, binary: Path, model: Path, config: dict[str, Any], timeout: float = 120):
        self.timeout = timeout
        self.directory = tempfile.TemporaryDirectory(prefix='muzero-session-')
        self.errors = tempfile.TemporaryFile(mode='w+t')
        self.process = None
        try:
            resolved = Path(self.directory.name) / 'eval.cfg'
            resolved.write_text(render_native_config(config))
            self.process = subprocess.Popen(
                [str(binary.resolve()), str(resolved), str(model.resolve()), '--serve'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
                text=True, bufsize=1,
            )
            self.canvas = self.receive()['canvas_size']
        except BaseException:
            self.close()
            raise

    def receive(self) -> dict[str, Any]:
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            self.close()
            raise RuntimeError('模型响应超时，请重新开始一局')
        line = self.process.stdout.readline()
        if not line:
            self.process.wait(timeout=5)
            self.errors.seek(0)
            message = self.errors.read()[-4000:].strip()
            raise RuntimeError(message or '模型进程已退出，请重新开始一局')
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            self.close()
            raise RuntimeError('模型返回了无效响应') from error
        if not response.get('ok'):
            raise RuntimeError(response.get('error', '模型请求失败'))
        return response

    def command(self, command: str) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError('模型进程已退出，请重新开始一局')
        try:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError('模型进程已断开，请重新开始一局') from error
        return self.receive()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        self.errors.close()
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
