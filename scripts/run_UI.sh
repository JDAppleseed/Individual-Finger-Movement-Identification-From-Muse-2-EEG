#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8008"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="5173"
HEALTH_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/health"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

require_cmd python
require_cmd lsof
require_cmd curl
require_cmd npm

pids_on_port() {
  local port="$1"
  lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null || true
}

wait_port_free() {
  local port="$1"
  local deadline=$((SECONDS + 5))
  while [ ${SECONDS} -lt ${deadline} ]; do
    if [ -z "$(pids_on_port "${port}")" ]; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

kill_port() {
  local port="$1"
  local pids
  pids="$(pids_on_port "${port}")"
  if [ -z "${pids}" ]; then
    return 0
  fi

  echo "Port ${port} in use by PID(s): ${pids}"
  for pid in ${pids}; do
    kill -TERM "${pid}" 2>/dev/null || true
  done

  if wait_port_free "${port}"; then
    return 0
  fi

  # Still not free: force kill
  for pid in ${pids}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done

  # Final check
  wait_port_free "${port}" || true
}

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  if [ -n "${BACKEND_PID}" ]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
  if [ -n "${FRONTEND_PID}" ]; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  # Don’t aggressively kill ports here; leave manual cleanup possible.
  echo "Stopped UI (backend+frontend)"
}
trap cleanup EXIT INT TERM

# 1) Free ports cleanly
kill_port "${BACKEND_PORT}"
kill_port "${FRONTEND_PORT}"

# 2) Start backend
mkdir -p demo_backend/logs
: > demo_backend/logs/ui_backend.log

# Bind explicitly to 127.0.0.1 so health checks match
python -m demo_backend.server \
  > demo_backend/logs/ui_backend.log 2>&1 &
BACKEND_PID=$!

# 3) Wait for backend health WITHOUT spamming curl errors
healthy=0
for _ in $(seq 1 60); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 0.2
done

if [ ${healthy} -ne 1 ]; then
  echo "Backend did not become healthy at ${HEALTH_URL}" >&2
  echo "--- tail demo_backend/logs/ui_backend.log ---" >&2
  tail -n 60 demo_backend/logs/ui_backend.log >&2 || true
  exit 1
fi

# 4) Start frontend
if [ ! -d demo_app/node_modules ]; then
  (cd demo_app && npm install)
fi

cd demo_app
: > ui_frontend.log
npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" \
  > ui_frontend.log 2>&1 &
FRONTEND_PID=$!
cd "${REPO_ROOT}"

echo "Backend:   ${HEALTH_URL}"
echo "Frontend:  http://localhost:${FRONTEND_PORT}/"
echo "Logs:      demo_backend/logs/ui_backend.log and demo_app/ui_frontend.log"

wait