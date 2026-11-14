#!/bin/sh
# Install the exact saved-checkpoint experiment used by the attitude notebook.
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_ROOT"
REVISION=eaf94a7c45c150e2d2f0b42dec204e82a3c14a66
SOURCE="$TASK_ROOT/build/attitude-legacy"
export UV_CACHE_DIR="$TASK_ROOT/build/uv-cache"
if [ ! -d "$SOURCE/.git" ]; then
    if [ -e "$SOURCE" ]; then
        echo "Refusing to overwrite existing $SOURCE" >&2
        exit 1
    fi
    git init "$SOURCE"
    git -C "$SOURCE" remote add origin https://github.com/bernardorivas/2027_L4DC_EquivariantMorseGraphs.git
    git -C "$SOURCE" sparse-checkout set code
    git -C "$SOURCE" fetch --depth=1 origin "$REVISION"
    git -C "$SOURCE" checkout --detach "$REVISION"
fi
if [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$REVISION" ] || [ -n "$(git -C "$SOURCE" status --porcelain --untracked-files=no)" ]; then
    echo "Legacy source differs from the pinned revision; preserve your edits and use a fresh build/attitude-legacy directory." >&2
    exit 1
fi
if command -v uv >/dev/null 2>&1; then
    if [ ! -x .venv-attitude/bin/python ]; then uv venv .venv-attitude --python 3.13; fi
    uv pip install --python .venv-attitude/bin/python -r "$SOURCE/code/requirements.txt"
else
    if [ ! -x .venv-attitude/bin/python ]; then python3.13 -m venv .venv-attitude; fi
    .venv-attitude/bin/python -m pip install -r "$SOURCE/code/requirements.txt"
fi
.venv-attitude/bin/python - "$SOURCE/code/examples" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from common.workflow import require_environment, verify_inputs
require_environment(graphics=True)
print('Verified legacy input files:', verify_inputs(Path(sys.argv[1]) / 'example3/periodic'))
PY
