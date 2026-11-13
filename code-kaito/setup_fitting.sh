#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_ROOT"
export UV_CACHE_DIR="$TASK_ROOT/build/uv-cache"
if command -v uv >/dev/null 2>&1; then
    if [ ! -x .venv-emlp/bin/python ]; then uv venv .venv-emlp --python 3.11; fi
    uv pip install --python .venv-emlp/bin/python -r requirements-emlp.txt
else
    if [ ! -x .venv-emlp/bin/python ]; then python3.11 -m venv .venv-emlp; fi
    .venv-emlp/bin/python -m pip install -r requirements-emlp.txt
fi
