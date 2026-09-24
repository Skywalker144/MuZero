#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TORCH_PREFIX="$("$PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S "$ROOT/cpp" -B "$ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
    -DMUZERO_WITH_TORCH=ON -DBUILD_TESTING=ON
cmake --build "$ROOT/build" --parallel "${BUILD_JOBS:-2}" "$@"
