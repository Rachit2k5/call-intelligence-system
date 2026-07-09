#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
VENV_PYTHON="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
echo
echo " Intelligent Call Prioritization System"
echo " Setting up (first run only) and starting the server..."
echo
if [ -x "$VENV_PYTHON" ]; then
  "$VENV_PYTHON" run.py
else
  python3 run.py
fi
