#!/usr/bin/env bash
# Starts the whole local SmartMonitoring stack (Redis, Celery worker, FL
# server) using the settings in .env, launches enough clients to satisfy
# FL_N against datasets/clients/, and opens the dashboard in the browser.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV="$ROOT_DIR/.venv"
PID_DIR="$ROOT_DIR/run"
mkdir -p "$PID_DIR"

# --- 0. Load .env -----------------------------------------------------------
ENV_FILE="$ROOT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "No .env found creating one from .env.example (edit it to change settings)."
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
fi
cp "$ENV_FILE" "$ROOT_DIR/server/.env"
cp "$ENV_FILE" "$ROOT_DIR/client/.env"

set -a
source "$ENV_FILE"
set +a

FL_N="${FL_N:-10}"
SERVER_HOST="${SERVER_HOST:-http://127.0.0.1:8000}"
SERVER_IP="${SERVER_IP:-127.0.0.1:8000}"
PASSWORD="${PASSWORD:-P7h1!quiBO0no96}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [ ! -d "$VENV" ]; then
  echo "No .venv found at $VENV."
  echo "Run first: python3 -m venv .venv && source .venv/bin/activate && pip install -r server/requirements.txt -r client/requirements.txt"
  exit 1
fi

is_running() {
  local pid_file="$1"
  [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

# --- 1. Redis -----------------------------------------------------------
if redis-cli ping > /dev/null 2>&1; then
  echo "Redis:   already running"
else
  echo "Redis:   starting..."
  redis-server --daemonize yes
  for _ in $(seq 1 10); do
    redis-cli ping > /dev/null 2>&1 && break
    sleep 0.5
  done
  redis-cli ping > /dev/null 2>&1 || { echo "Redis failed to start."; exit 1; }
fi

# Only flush if nothing is currently running — a stale fl:coordinator_lock
# (acquired once per round in server/main.py and never released) otherwise
# blocks a fresh server from ever starting a round again. Skipped when the
# server/celery are already up so re-running this script doesn't nuke a
# session in progress.
if ! is_running "$PID_DIR/server.pid" 2>/dev/null; then
  echo "Redis:   flushing stale state for a clean start..."
  redis-cli flushall > /dev/null
fi

# --- 2. Build the dashboard if it hasn't been built yet -----------------
if [ ! -f "$ROOT_DIR/server/static/index.html" ]; then
  if command -v npm > /dev/null 2>&1; then
    echo "Dashboard: building (first run)..."
    (cd "$ROOT_DIR/dashboard" && npm install --silent && npm run build)
  else
    echo "Dashboard: npm not found, skipping build — the API will still work but '/' won't serve the UI."
  fi
else
  echo "Dashboard: already built"
fi

# --- 3. Celery worker -----------------------------------------------------
if is_running "$PID_DIR/celery.pid"; then
  echo "Celery:  already running (pid $(cat "$PID_DIR/celery.pid"))"
else
  echo "Celery:  starting..."
  (
    cd "$ROOT_DIR/server"
    source "$VENV/bin/activate"
    exec celery -A celery_app.celery_app worker --loglevel=info
  ) > "$ROOT_DIR/celery.log" 2>&1 &
  echo $! > "$PID_DIR/celery.pid"
  disown
  sleep 3
fi

# --- 4. FL server (uvicorn) ------------------------------------------------
if is_running "$PID_DIR/server.pid"; then
  echo "Server:  already running (pid $(cat "$PID_DIR/server.pid"))"
else
  echo "Server:  starting..."
  (
    cd "$ROOT_DIR/server"
    source "$VENV/bin/activate"
    exec python -m uvicorn main:app --host "$HOST" --port "$PORT"
  ) > "$ROOT_DIR/server.log" 2>&1 &
  echo $! > "$PID_DIR/server.pid"
  disown
fi

echo -n "Server:  waiting for it to come up"
for _ in $(seq 1 40); do
  if curl -s -o /dev/null -w "%{http_code}" "$SERVER_HOST/api/status" 2>/dev/null | grep -q "200"; then
    echo " — up."
    break
  fi
  echo -n "."
  sleep 1
done

# --- 5. Launch clients to satisfy FL_N, from datasets/clients/ -------------
AVAILABLE=$(ls "$ROOT_DIR/datasets/clients"/client_*.npz 2>/dev/null | wc -l)
N_TO_LAUNCH=$FL_N
if [ "$N_TO_LAUNCH" -gt "$AVAILABLE" ]; then
  echo "Clients: FL_N=$FL_N but only $AVAILABLE dataset file(s) in datasets/clients/ — launching $AVAILABLE."
  N_TO_LAUNCH=$AVAILABLE
fi

mkdir -p "$ROOT_DIR/client/run_logs"
: > "$PID_DIR/clients.pid"
echo "Clients: launching $N_TO_LAUNCH (client_0..client_$((N_TO_LAUNCH - 1)))..."
for i in $(seq 0 $((N_TO_LAUNCH - 1))); do
  (
    cd "$ROOT_DIR/client"
    source "$VENV/bin/activate"
    exec python main.py \
      -d "../datasets/clients/client_${i}.npz" \
      -s "$SERVER_IP" \
      -p "$PASSWORD" \
      -c "client_${i}"
  ) > "$ROOT_DIR/client/run_logs/client_${i}.log" 2>&1 &
  echo $! >> "$PID_DIR/clients.pid"
  disown
done

# --- 6. Open the dashboard --------------------------------------------------
echo "Dashboard: opening $SERVER_HOST/ ..."
if command -v xdg-open > /dev/null 2>&1; then
  DISPLAY="${DISPLAY:-:1}" xdg-open "$SERVER_HOST/" > /dev/null 2>&1 &
  disown
else
  echo "  (xdg-open not found — open $SERVER_HOST/ manually)"
fi

echo ""
echo "All started. Logs: server.log, celery.log, client/run_logs/client_*.log"
echo "Stop everything with: ./stop.sh"
