#!/usr/bin/env bash
# One-time setup on macOS or Linux: Python environment, packages, .env and models.
#
#   bash setup.sh                 # everything, including the ~20 GB of models
#   bash setup.sh --skip-models   # models download on first use instead
set -euo pipefail
cd "$(dirname "$0")"

# The pinned PyTorch has no builds for Python 3.13 or newer.
supported() {
  "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' >/dev/null 2>&1
}

if [ -x .venv/bin/python ]; then
  if ! supported .venv/bin/python; then
    echo "The existing .venv uses an unsupported Python version. Delete the .venv folder and run this again."
    exit 1
  fi
else
  PYTHON=""
  for candidate in python3.11 python3.12 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && supported "$candidate"; then
      PYTHON="$candidate"
      break
    fi
  done
  if [ -z "$PYTHON" ]; then
    echo "Python 3.10, 3.11 or 3.12 is required (3.11 is tested); none was found."
    echo "Install Python 3.11 from https://www.python.org/downloads/ and run this again."
    exit 1
  fi
fi

# llama-cpp-python is compiled during installation.
if [ "$(uname)" = "Darwin" ]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "The Xcode Command Line Tools are required to compile llama-cpp-python."
    echo "Run: xcode-select --install   then run this again."
    exit 1
  fi
elif ! command -v c++ >/dev/null 2>&1; then
  echo "A C/C++ compiler is required to compile llama-cpp-python."
  echo "For example, on Ubuntu or Debian: sudo apt install build-essential   then run this again."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "==> Creating the Python environment (.venv) with $("$PYTHON" --version)"
  "$PYTHON" -m venv .venv
fi

echo "==> Installing packages (compiling llama-cpp-python can take several minutes)"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
fi

if [ "${1:-}" = "--skip-models" ]; then
  echo "==> Skipping model downloads: each model downloads the first time it is needed"
else
  echo "==> Downloading the models (about 20 GB)"
  .venv/bin/python download_models.py
fi

echo
echo "Setup complete. Start the app with: bash start.sh"
