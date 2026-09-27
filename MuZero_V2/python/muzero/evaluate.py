import argparse
from pathlib import Path
import subprocess
import tempfile

from .config import ROOT, load_eval_config, render_native_config
from .protocol import SPEC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--config-dir', default='configs/baseline')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/muzero_eval')
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--rule', choices=tuple(SPEC['rules']), default='freestyle')
    parser.add_argument('--moves', type=int, nargs='*', default=[])
    args = parser.parse_args()
    config = load_eval_config(args.config_dir)
    with tempfile.TemporaryDirectory(prefix='muzero-eval-') as tmp:
        resolved = Path(tmp) / 'resolved.cfg'
        resolved.write_text(render_native_config(config))
        result = subprocess.run([str(args.binary.resolve()), str(resolved), str(args.model.resolve()),
                                 str(args.size), args.rule, *map(str, args.moves)])
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
