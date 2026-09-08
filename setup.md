# Local Setup & Testing Guide — SmartMonitoring Federated Learning

This guide walks through setting up a **fully local, offline** environment to run the SmartMonitoring
Federated Learning system with all **10 clients** in [datasets/clients/](datasets/clients/)
(`client_0.npz` … `client_9.npz`), using the mock S3 storage backend so no AWS account is required.

No source code changes are needed — this only covers environment setup and run commands.

---

## 1. What you're setting up

| Component | Role | Port |
|---|---|---|
| Redis | Broker for Celery + shared state/pub-sub for the FL coordinator | 6379 |
| Celery worker | Runs model aggregation (FedAvg etc.) as background tasks | — |
| FastAPI/uvicorn server (`server/main.py`) | Auth, WebSocket coordinator, mock S3 endpoints | 8000 |
| 10x Client processes (`client/main.py`) | Each trains on one `datasets/clients/client_N.npz` file | — |

`datasets/clients/` has exactly 10 files, and [server/credentials.json](server/credentials.json)
already has entries for `client_0`–`client_9` (and many more), all hashed from the password
`P7h1!quiBO0no96` (the same default in [.env.example](.env.example)) — so no credential edits are needed
to test these 10 clients out of the box. The server's default `FL_N=10` also matches the 10 client
dataset files exactly.

---

## 2. Prerequisites (OS packages)

You're on Ubuntu 24.04. Install Redis and basic build tools system-wide (this is the only step that
needs `sudo`; everything else is isolated in a Python virtual environment):

```bash
sudo apt update
sudo apt install -y redis-server python3-venv python3-pip
```

Verify Redis:
```bash
redis-server --version
```

Python 3.12 (already installed on this machine) is fine — the project doesn't pin a specific version.

---

## 3. Create an isolated virtual environment

Since the server and client both run on the same machine for local testing, one shared venv is
simplest (it's a superset of both dependency lists):

```bash
cd /home/pai/Downloads/SmartMonitoring
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Keep this venv activated (`source .venv/bin/activate`) in every terminal you use for the rest of this
guide.

---

## 4. Install project dependencies

```bash
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

For running the test suite in [tests/](tests/) (it uses FastAPI's `TestClient`, which needs `httpx`,
plus `pytest` as a runner):

```bash
pip install pytest httpx
```

> Note: `tensorflow` and `codecarbon` are large installs — expect this step to take several minutes and
> a few GB of disk space.

---

## 5. Configure environment variables

Copy the template and edit it for local/mock testing:

```bash
cp .env.example .env
```

Edit `.env` so it matches a fully local run (key changes from the template highlighted):

```bash
FL_N=10              # exactly matches the 10 datasets/clients/*.npz files
FL_K=3                # how many clients are sampled per round (tune as desired, max 10)
FL_ROUNDS=5           # keep small for a quick local smoke test
FL_AGGREGATOR=fedavg
FL_SELECTOR=rl

SERVER_HOST=http://127.0.0.1:8000
HOST=0.0.0.0
PORT=8000
REDIS_URL=redis://localhost:6379/0

S3_MOCK=true          # IMPORTANT: avoids needing real AWS credentials/S3 bucket

CLIENT_ID=client_0
PASSWORD=P7h1!quiBO0no96
SERVER_IP=127.0.0.1:8000
```

Both `server/main.py` and `client/main.py` auto-load `.env` via `python-dotenv` if present, but since
they each run from their own working directory (`server/` and `client/`), the simplest approach is to
also place a copy of `.env` in each subdirectory:

```bash
cp .env server/.env
cp .env client/.env
```

`server/credentials.json` already contains valid password hashes for `client_0`–`client_9` — no edits
needed there for this test.

---

## 6. Start Redis

```bash
redis-server --daemonize yes
redis-cli ping   # should print PONG
```

(If you'd rather see logs live, skip `--daemonize` and run it in its own terminal instead.)

---

## 7. Start the Celery worker

In a new terminal (with the venv activated):

```bash
cd /home/pai/Downloads/SmartMonitoring/server
source ../.venv/bin/activate
celery -A celery_app.celery_app worker --loglevel=info
```

Leave this running.

---

## 8. Start the FastAPI/uvicorn server

In another new terminal:

```bash
cd /home/pai/Downloads/SmartMonitoring/server
source ../.venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

On first startup it will create and mock-upload an initial `global_model_0.keras` (you'll see
"Initial global model not found in S3. Creating and uploading..." in the logs), and it resets Redis's
FL run-state keys. Leave this running — it's your FL coordinator.

---

## 9. Launch the 10 clients against `datasets/clients/`

In a final terminal, run all 10 clients, each pointed at its own dataset file and matching client ID.
Since each `client/main.py` invocation handles exactly one client/dataset pair, launch them as
background processes from the `client/` directory (client scripts write local files like `psswd.txt`,
`models/`, `metrics/` relative to their CWD, so running them all from `client/` is intentional and
matches how the project is normally used):

```bash
cd /home/pai/Downloads/SmartMonitoring/client
source ../.venv/bin/activate

for i in $(seq 0 9); do
  python main.py \
    -d ../datasets/clients/client_${i}.npz \
    -s 127.0.0.1:8000 \
    -p P7h1!quiBO0no96 \
    -c client_${i} \
    > client_${i}.log 2>&1 &
done

wait   # blocks until all 10 client processes finish (or Ctrl+C to stop watching)
```

Each client:
1. Authenticates via challenge/response against the server.
2. Opens a WebSocket at `ws://127.0.0.1:8000/ws/client_N`.
3. Waits until all 10 clients have connected (`FL_N=10`) — this is what triggers the coordinator to
   start the first round.
4. Trains when selected (`FL_K` clients per round), evaluates every round, and produces per-client
   plots (`client/metrics/loss_vs_round_client_client_N.png`, etc.) once all `FL_ROUNDS` complete.

Watch progress with:
```bash
tail -f client_0.log        # per-client log
```
and in the uvicorn terminal you'll see round-by-round selection/aggregation logs.

---

## 10. Where to check results

- `server/global_metrics.txt` — appended global round metrics (loss/accuracy/F1/latency/energy).
- `server/loss_vs_round.png`, `accuracy_vs_round.png`, `f1_vs_round.png`,
  `system_resources_vs_round.png` — server-side plots per round (also mock-uploaded to
  `server/tmp_s3_bucket/plots/`).
- `client/metrics/*_client_client_N.png` — per-client local vs. global convergence plots.
- `server/tmp_s3_bucket/` — the mock "S3 bucket": global models per round
  (`models/global/global_model_R.keras`) and each client's uploaded weights
  (`models/round_R/client_N.keras`).
- `server/upload_log.csv` — a log of every client upload per round.

A full run finishes when `FL_ROUNDS` rounds complete; the server then broadcasts an `exit` command and
each client process prints `Finished Training and Evaluation.` and exits.

---

## 11. Running the automated test suite

The tests in [tests/](tests/) also talk to a real Redis instance (they import `server/main.py`
directly), so keep Redis running (step 6) before running them. They do **not** require the Celery
worker or uvicorn server to be running, and they do not touch `datasets/clients/`.

```bash
cd /home/pai/Downloads/SmartMonitoring
source .venv/bin/activate
PYTHONPATH=server:client pytest tests/ -v
```

If a test imports `main` from `server/` and hangs or errors on Redis connection, double check
`redis-cli ping` succeeds first.

---

## 12. Stopping everything / cleanup

```bash
# Stop client processes if still running
pkill -f "python main.py"

# Stop uvicorn and celery worker: Ctrl+C in their terminals

# Stop redis (only if you started it standalone, not via systemd)
redis-cli shutdown
```

To reset state for a fresh run, clear generated artifacts (safe to delete, all reproducible):
```bash
rm -rf server/tmp_s3_bucket server/upload_log.csv server/global_metrics.txt \
       server/*.png server/models client/models client/metrics client/*.log \
       client/psswd.txt
redis-cli flushall
```

---

## Troubleshooting

- **Port 8000/6379 already in use**: another process is bound to it — `lsof -i :8000` /
  `lsof -i :6379` to find and stop it, or change `PORT`/`REDIS_URL`.
- **Clients hang on "Not selected." forever**: expected for non-selected clients each round — they
  still get pinged to evaluate every round; only training is limited to `FL_K` clients.
- **Coordinator never starts**: it only launches once `FL_N` clients are simultaneously connected via
  WebSocket — check `server/credentials.json` has all `client_0`..`client_9` entries and all 10
  client processes authenticated successfully (check each `client_N.log`).
- **`ModuleNotFoundError` for `httpx`**: needed only for the pytest suite (FastAPI's `TestClient`) —
  `pip install httpx`.
- **High RAM usage**: TensorFlow + 10 concurrent client processes can be memory-heavy; if the machine
  struggles, launch clients in smaller batches (e.g. 5 at a time) instead of all 10 with `&` at once —
  note the coordinator still won't start training until all 10 have connected, since `FL_N=10`.
