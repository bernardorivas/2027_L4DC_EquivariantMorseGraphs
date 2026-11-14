#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PATH="$TASK_ROOT/.venv/bin:$PATH"
export MPLCONFIGDIR="$TASK_ROOT/build/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/build/cache"
export IPYTHONDIR="$TASK_ROOT/build/ipython"
export JUPYTER_CONFIG_DIR="$TASK_ROOT/build/jupyter/config"
export JUPYTER_DATA_DIR="$TASK_ROOT/build/jupyter/data"
export JUPYTER_RUNTIME_DIR="$TASK_ROOT/build/jupyter/runtime"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
cd "$TASK_ROOT"
NOTEBOOK="${1:-notebooks/01_sphere.ipynb}"
if [ "$#" -gt 0 ]; then shift; fi
exec "$TASK_ROOT/.venv/bin/python" -m jupyterlab --ServerApp.ip=127.0.0.1 --ServerApp.root_dir="$TASK_ROOT" "$NOTEBOOK" "$@"
