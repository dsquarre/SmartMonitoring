# Running SmartMonitoring — Server, Clients & Dashboard

This covers two ways to run the project locally: the **automated** way
(`start.sh` / `stop.sh`, one command) and the **manual** way (each piece in
its own terminal, useful for debugging one component at a time).

Both assume the one-time setup below is already done — see
[setup.md](setup.md) for full details (installing Redis, creating the
virtualenv, etc.). The short version:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt -r client/requirements.txt
cp .env.example .env   # then edit FL_N / FL_K / FL_ROUNDS / S3_MOCK etc. as needed
```

The dashboard (`dashboard/`) is a separate React app that gets **built**
into `server/static/` and served by the FastAPI server at `/`. Build it
once with:

```bash
cd dashboard
npm install
npm run build
```

(`start.sh` also does this automatically on first run if `server/static/`
doesn't exist yet.)

---

## Option A — Automated: `start.sh` / `stop.sh`

The fastest way to get everything running end to end.

```bash
./start.sh
```

What it does, in order:

1. Loads `.env` from the repo root (creates it from `.env.example` if
   missing) and syncs it into `server/.env` and `client/.env`.
2. Starts Redis if it isn't already running, and flushes it for a clean
   session — **but only if the server isn't already running** (safe to
   re-run `start.sh` against a session already in progress).
3. Builds the dashboard (`dashboard/` → `server/static/`) if it hasn't
   been built yet.
4. Starts the Celery worker (aggregation backend) in the background,
   logging to `celery.log`.
5. Starts the FastAPI/uvicorn server in the background, logging to
   `server.log`, and waits for it to respond.
6. Launches enough client processes to satisfy `FL_N` from `.env`
   against `datasets/clients/client_0.npz … client_9.npz` (capped at however
   many `.npz` files actually exist), logging each to
   `client/run_logs/client_N.log`.
7. Opens `http://127.0.0.1:8000/` in your browser automatically.

From there the FL round(s) run on their own — watch progress live in the
dashboard, or tail the log files.

To stop everything:

```bash
./stop.sh
```

This kills the client processes, the FastAPI server, and the Celery
worker. **Redis is left running** as a system service (it's shared
infrastructure, not something this project owns) — stop it yourself with
`sudo systemctl stop redis-server` if you want it down too.

PID files live in `run/` (gitignored); logs are `server.log`, `celery.log`,
and `client/run_logs/client_*.log` (all gitignored).

### Re-running

`start.sh` is safe to run again:
- If the server/Celery worker are already up, it reuses them instead of
  starting duplicates.
- If they're *not* running, it flushes Redis first — this matters because
  the FL coordinator takes a Redis lock each round that the server code
  never explicitly releases, so a stale lock from a previous session would
  otherwise silently block a fresh server from ever starting a round. This
  is a pre-existing quirk in `server/main.py`'s coordinator, worked around
  here rather than patched, since it doesn't affect a single continuous
  run — only back-to-back fresh starts against the same Redis instance.

### Changing what gets run

Edit `.env` before running `start.sh` — e.g. lower `FL_ROUNDS` for a quick
smoke test, or set `FL_N` to however many of the 10 client datasets you
want to include. `start.sh` always re-reads `.env` on each invocation.

---

## Option B — Manual: one terminal per component

Useful when you want to see one component's logs directly, restart just
one piece, or step through what's happening.

**Terminal 1 — Redis** (skip if already running as a service):
```bash
redis-server
```

**Terminal 2 — Celery worker:**
```bash
cd server
source ../.venv/bin/activate
celery -A celery_app.celery_app worker --loglevel=info
```

**Terminal 3 — FastAPI server** (also serves the dashboard at `/`):
```bash
cd server
source ../.venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Once this is up, open **http://localhost:8000/** in your browser — that's
the dashboard. It'll show "Waiting for clients to connect — 0/N" until
enough clients join.

**Terminal 4+ — one client per terminal** (repeat per client, using its own
dataset file and matching `client_id` from `server/credentials.json`):
```bash
cd client
source ../.venv/bin/activate
python main.py -d ../datasets/clients/client_0.npz -s 127.0.0.1:8000 -p 'P7h1!quiBO0no96' -c client_0
```

Or launch several in the background from one terminal:
```bash
cd client
source ../.venv/bin/activate
for i in 0 1 2 3 4; do
  python main.py -d ../datasets/clients/client_${i}.npz -s 127.0.0.1:8000 -p 'P7h1!quiBO0no96' -c client_${i} \
    > run_logs/client_${i}.log 2>&1 &
done
```

The FL coordinator starts automatically once `FL_N` clients (from `.env`)
have connected — watch it happen live in the dashboard, or in the uvicorn
terminal's logs.

**Stopping manually:** `Ctrl+C` each terminal (server, celery, each
client), or from another terminal:
```bash
pkill -f "python main.py"
pkill -f "uvicorn main:app"
pkill -f "celery -A celery_app"
```

---

## The dashboard

Whichever way you start the server, the dashboard is always at
**http://localhost:8000/** — it's served directly by the FastAPI app (see
`server/static/`), not a separate dev server. It polls a handful of
read-only endpoints (`/api/status`, `/api/clients`, `/api/metrics/history`,
`/plots/*.png`) every couple of seconds, so just leave the tab open and it
updates on its own — no manual refresh needed.

If you edit the dashboard source (`dashboard/src/`), rebuild it and reload
the page:
```bash
cd dashboard
npm run build
```

### What each panel means

- **System Health** — green/red dot per connection: FL server itself,
  Redis, and S3 storage (mock or real AWS, per `S3_MOCK` in `.env`).
- **Federated Round** — current round / total, clients selected this round
  vs `FL_K`, and a shimmering "Aggregating…" banner exactly while the
  Celery worker is running FedAvg (or whichever aggregator is configured).
- **Connected Clients** — one card per connected client, cycling through
  Connected → Training… → Uploaded → Evaluated as the round progresses,
  with brief download/upload arrow animations on state changes.
- **Training Metrics** — the latest round's numbers plus the four
  server-generated plots (loss, accuracy, F1, system resources), which
  refresh automatically whenever a new round completes.

---

## Troubleshooting

- **Coordinator never starts / dashboard stuck on "Waiting for clients"**:
  confirm `FL_N` clients actually connected — check `connected_clients` in
  the dashboard's health panel, or `client/run_logs/client_N.log` for
  auth/connection errors. Also see the stale-lock note under Option A if
  you're restarting against a Redis instance from a previous run.
- **Dashboard shows "Can't reach the FL server"**: the uvicorn process
  isn't up yet or crashed — check `server.log`.
- **`/` returns something other than the dashboard**: `server/static/`
  hasn't been built — run `cd dashboard && npm install && npm run build`.
- **Port 8000 or 6379 already in use**: another process is bound to it —
  `lsof -i :8000` / `lsof -i :6379` to find it, or change `PORT` /
  `REDIS_URL` in `.env`.
