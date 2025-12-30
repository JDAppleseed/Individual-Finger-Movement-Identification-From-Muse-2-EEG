#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/demo_backend/server.py" &
BACKEND_PID=$!

echo "Backend running (pid $BACKEND_PID)."

cd "$ROOT_DIR/demo_app"
if [ ! -d node_modules ]; then
  npm install
fi
npm run dev

kill $BACKEND_PID
