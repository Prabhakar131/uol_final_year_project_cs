#!/usr/bin/env bash
# Starts CloudIR Trainer and opens it in the browser. Run setup.sh once first.
#
#   bash start.sh
#   CLOUDIR_PORT=5050 bash start.sh   # if port 5000 is busy
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ] || [ ! -f .env ]; then
  echo "Run 'bash setup.sh' first."
  exit 1
fi

export CLOUDIR_PORT="${CLOUDIR_PORT:-5000}"
# 127.0.0.1, not localhost: on macOS, AirPlay Receiver answers localhost:5000.
URL="http://127.0.0.1:$CLOUDIR_PORT"

open_browser() {
  open "$URL" 2>/dev/null || xdg-open "$URL" >/dev/null 2>&1 || true
}

if curl -fs "$URL/api/health" 2>/dev/null | grep -q "CloudIR Trainer"; then
  echo "CloudIR Trainer is already running at $URL"
  open_browser
  exit 0
fi

# Open the browser once the server answers.
(
  for _ in $(seq 1 120); do
    if curl -fs "$URL/api/health" >/dev/null 2>&1; then
      open_browser
      exit 0
    fi
    sleep 1
  done
) &

echo "Starting CloudIR Trainer at $URL (press Ctrl+C to stop)"
echo "If port $CLOUDIR_PORT is busy, run: CLOUDIR_PORT=5050 bash start.sh"
exec .venv/bin/python app.py
