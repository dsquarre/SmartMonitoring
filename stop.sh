#!/usr/bin/env bash
# Stops everything started by start.sh (clients, FL server, Celery worker).
# Redis is left running as a system service; stop it yourself with
# `sudo systemctl stop redis-server` if you want it down too.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$ROOT_DIR/run"

kill_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null
      echo "$label: stopped (pid $pid)"
    fi
    rm -f "$pid_file"
  fi
}

echo "Stopping clients..."
if [ -f "$PID_DIR/clients.pid" ]; then
  while read -r pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  done < "$PID_DIR/clients.pid"
  rm -f "$PID_DIR/clients.pid"
fi
pkill -9 -f "main.py -d " 2>/dev/null

kill_pid_file "$PID_DIR/server.pid" "Server"
pkill -9 -f "uvicorn main:app" 2>/dev/null

kill_pid_file "$PID_DIR/celery.pid" "Celery"
pkill -9 -f "celery -A celery_app.celery_app worker" 2>/dev/null

echo "All stopped."
