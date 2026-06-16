#!/usr/bin/env bash
# Launch the ollama-asr TUI, creating the venv on first run.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
if [[ ! -d "$VENV" ]]; then
    echo "Creating virtualenv and installing dependencies…"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r requirements.txt
fi

exec "$VENV/bin/python" app.py "$@"
