import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from muzero.config import ROOT, load_config, load_eval_config
from .app import App, Conflict


STATIC = Path(__file__).resolve().parent / 'static'


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status: int, data: bytes, content_type: str):
            try:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, status: int, value):
            self.send(status, json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/api/state':
                try:
                    since = int(parse_qs(url.query).get('since', ['-1'])[0])
                except ValueError:
                    return self.json(400, dict(error='无效版本号'))
                return self.json(200, app.snapshot(since))
            if url.path == '/api/catalog':
                return self.json(200, app.catalog())
            files = {'/': 'index.html', '/app.js': 'app.js', '/styles.css': 'styles.css'}
            if url.path in files:
                path = STATIC / files[url.path]
                return self.send(200, path.read_bytes(), (mimetypes.guess_type(path)[0] or 'text/plain') + '; charset=utf-8')
            self.json(404, dict(error='页面不存在'))

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                return self.json(403, dict(error='请求来源不匹配'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('无效请求长度')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('请求必须是对象')
                operations = {'/api/new': 'new', '/api/play': 'play', '/api/undo': 'undo', '/api/retry': 'retry'}
                if self.path not in operations:
                    return self.json(404, dict(error='未知接口'))
                self.json(202, app.submit(operations[self.path], payload))
            except Conflict as error:
                self.json(409, dict(error=str(error)))
            except (ValueError, TypeError) as error:
                self.json(400, dict(error=str(error)))

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description='MuZero 本地对弈')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config-dir', default='configs/exp_baseline')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/muzero_eval')
    args = parser.parse_args()
    paths = sorted(args.models_dir.glob('**/models/model_*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    models = {str(path.relative_to(args.models_dir)): path.resolve() for path in paths}
    if args.model:
        selected = args.model.resolve()
        if not selected.is_file():
            parser.error('指定的模型文件不存在')
        key = next((key for key, path in models.items() if path == selected), str(selected))
        models = {key: selected, **{key: path for key, path in models.items() if path != selected}}
    if not models:
        parser.error('未找到导出的模型，请用 --model 指定 TorchScript 模型')
    if not args.binary.is_file():
        parser.error('缺少 muzero_eval，请先运行 web/webui.sh 构建')
    config = load_eval_config(args.config_dir)
    sizes = load_config(args.config_dir, environ={})['BOARD_SIZES']
    app = App(args.binary, models, config, max(sizes))
    server = make_server(app, args.host, args.port)
    print(f'MuZero Web: http://{args.host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == '__main__':
    main()
