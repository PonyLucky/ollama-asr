#!/usr/bin/env bash
# Launch the ollama-asr TUI, creating the venv on first run.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

VENV=".venv"
if [[ ! -d "$VENV" ]]; then
    echo "Creating virtualenv and installing dependencies…"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r requirements.txt
fi
PYTHON="$DIR/$VENV/bin/python"

# Open the TUI in the first available terminal emulator, passing along any extra
# args (e.g. --autostart). Used by --toggle when no app is running yet.
open_in_terminal() {
    local term
    for term in kitty alacritty foot konsole xterm tilix; do
        if command -v "$term" >/dev/null 2>&1; then
            setsid "$term" -e "$PYTHON" "$DIR/app.py" "$@" >/dev/null 2>&1 &
            return 0
        fi
    done
    if command -v wezterm >/dev/null 2>&1; then
        setsid wezterm start -- "$PYTHON" "$DIR/app.py" "$@" >/dev/null 2>&1 &
        return 0
    fi
    for term in gnome-terminal xfce4-terminal; do
        if command -v "$term" >/dev/null 2>&1; then
            setsid "$term" -- "$PYTHON" "$DIR/app.py" "$@" >/dev/null 2>&1 &
            return 0
        fi
    done
    return 1
}

# --toggle: bind this to your global shortcut. If the app is already running it
# toggles recording; otherwise it opens the TUI in a terminal and starts
# recording right away, so one key both launches and records. The argument is
# passed straight through to app.py (same flag name on both sides), so adding or
# renaming app.py's flags later needs no change here.
if [[ "${1:-}" == "--toggle" ]]; then
    if "$PYTHON" "$DIR/app.py" "$@" 2>/dev/null; then
        exit 0  # an app was already running and got toggled
    fi
    if ! open_in_terminal --autostart; then
        echo "ollama-asr: no supported terminal emulator found to launch the app." >&2
        exit 1
    fi
    exit 0
fi

exec "$PYTHON" "$DIR/app.py" "$@"
