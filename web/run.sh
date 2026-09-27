#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
conda run --no-capture-output -n pytorch bash MuZero_V2/scripts/build.sh --target muzero_eval
export PYTHONPATH="$ROOT/MuZero_V2/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n pytorch python -m web.server "$@"
