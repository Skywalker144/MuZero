#!/usr/bin/env bash
set -euo pipefail

session_name='gc'

if ! command -v tmux >/dev/null 2>&1; then
    printf '%s\n' 'Error: tmux is not installed or not in PATH.' >&2
    exit 1
fi

if tmux has-session -t "$session_name" 2>/dev/null; then
    printf "Error: tmux session '%s' already exists.\n" "$session_name" >&2
    exit 1
fi

exec tmux new-session -s "$session_name" "exec codex -m gpt-6-luna -c 'model_reasoning_effort=\"high\"' '\$git-commit'"
