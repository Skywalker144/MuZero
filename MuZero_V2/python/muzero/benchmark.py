import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .config import ROOT, load_config, render_native_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/muzero_benchmark')
    parser.add_argument('--config-dir', default='configs/baseline')
    parser.add_argument('--device')
    parser.add_argument('--threads', type=int, nargs='+')
    parser.add_argument('--batch-sizes', type=int, nargs='+')
    parser.add_argument('--wait-us', type=int, nargs='+')
    parser.add_argument('--searches-per-thread', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config_dir)
    if args.device:
        config['DEVICE'] = args.device
    threads = args.threads or [config['NUM_GAME_THREADS']]
    batches = args.batch_sizes or [config['NN_MAX_BATCH_SIZE']]
    waits = args.wait_us or [config['NN_BATCH_WAIT_US']]
    if min(threads + batches + [args.searches_per_thread, args.repeats]) < 1 or min(waits) < 0:
        parser.error('Counts must be positive and waits nonnegative')
    binary, model = args.binary.resolve(), args.model.resolve()
    provenance = {
        'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'model': str(model), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='muzero-benchmark-') as tmp, args.output.open('x') as output:
        resolved = Path(tmp) / 'resolved.cfg'
        for repeat in range(args.repeats):
            for count, batch, wait in itertools.product(threads, batches, waits):
                selected = dict(config, NUM_GAME_THREADS=count, NN_MAX_BATCH_SIZE=batch, NN_BATCH_WAIT_US=wait)
                resolved.write_text(render_native_config(selected))
                result = subprocess.run([str(binary), str(resolved), str(model), str(args.searches_per_thread)],
                                        capture_output=True, text=True)
                if result.returncode:
                    print(result.stderr, file=sys.stderr, end='')
                    result.check_returncode()
                metrics = json.loads(result.stdout)
                expected = count * args.searches_per_thread * (selected['FULL_SEARCH_VISITS'] + 1)
                if metrics['requests'] != expected:
                    raise ValueError(f'Incomplete search workload: {metrics["requests"]} != {expected}')
                row = dict(provenance, repeat=repeat, config=selected, metrics=metrics)
                output.write(json.dumps(row, sort_keys=True) + '\n')
                output.flush()
                print(json.dumps(dict(repeat=repeat, **metrics)), flush=True)


if __name__ == '__main__':
    main()
