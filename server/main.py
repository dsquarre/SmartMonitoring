import os
import csv
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import tensorflow as tf
from model import Model
from collections import deque
import random
import numpy as np
import time
from selector import RandomClientSelector, RLClientSelector, RandomRLAgent, QLearningAgent
from aggregator import FedAvg, FedFV, qFedAvg, FedAdam
from rl_env import FederatedEnv
import hashlib
import hmac

import redis
import redis.asyncio as aioredis
import boto3
from s3_helper import (
    generate_presigned_download_url,
    generate_presigned_upload_url,
    upload_file,
    get_bucket_name
)
from celery_app import aggregate_models_task

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration variables
N = int(os.environ.get("FL_N", "10"))
K = int(os.environ.get("FL_K", "3"))
ROUNDS = int(os.environ.get("FL_ROUNDS", "2"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Loop-safe Async Redis Proxy to handle TestClient's loop closures
class AsyncRedisProxy:
    def __init__(self, url):
        self.url = url
        self.client = None
        self.loop = None

    def get_client(self):
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if self.client is None or self.loop is not current_loop or (self.loop and self.loop.is_closed()):
            self.client = aioredis.from_url(self.url, protocol=2, decode_responses=True)
            self.loop = current_loop
        return self.client

    def __getattr__(self, name):
        return getattr(self.get_client(), name)

# Redis clients
redis_sync = redis.from_url(REDIS_URL, protocol=2, decode_responses=True)
redis_async = AsyncRedisProxy(REDIS_URL)

app = FastAPI()

# Ensure S3 Bucket structure is initialized on startup
@app.on_event("startup")
async def startup_event():
    # Reset transient state in Redis
    await redis_async.delete("fl:active_clients")
    await redis_async.set("fl:is_running", "false")
    await redis_async.set("fl:current_round", "0")
    await redis_async.set("fl:rounds_left", str(ROUNDS))
    await redis_async.delete("fl:round_history")
    
    # Initialize static client profiles in Redis if empty
    profiles_exist = await redis_async.exists("fl:client_profiles")
    if not profiles_exist:
        for i in range(100):
            profile = {
                "cpu_frequency": float(random.choice([1.2e9, 1.6e9, 2.0e9, 2.4e9, 2.8e9])),
                "tx_power": float(random.choice([0.1, 0.2, 0.3, 0.4, 0.5])),
                "r_trans": float(random.choice([5e6, 10e6, 15e6, 20e6, 30e6]))
            }
            await redis_async.hset("fl:client_profiles", str(i), json.dumps(profile))

    # Verify initial global model exists in S3, if not build and upload
    bucket = get_bucket_name()
    from s3_helper import S3_MOCK, MOCK_S3_DIR
    global_model_exists = False
    
    if S3_MOCK:
        mock_path = os.path.join(MOCK_S3_DIR, "models/global/global_model_0.keras")
        global_model_exists = os.path.exists(mock_path)
    else:
        s3_client = boto3.client("s3")
        try:
            s3_client.head_object(Bucket=bucket, Key="models/global/global_model_0.keras")
            global_model_exists = True
        except Exception:
            global_model_exists = False

    if global_model_exists:
        print("Found initial global model in S3.")
    else:
        print("Initial global model not found in S3. Creating and uploading...")
        os.makedirs("models", exist_ok=True)
        model = Model()
        model_path = "models/global_model_0.keras"
        model.model.save(model_path)
        upload_file(model_path, "models/global/global_model_0.keras")
        if os.path.exists(model_path):
            os.remove(model_path)
        try:
            os.rmdir("models")
        except Exception:
            pass

if not os.path.exists('upload_log.csv'):
    with open('upload_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "client_id", "filename", "weight", "round"])

def log_upload(client_id, filename, weight, round_num):
    with open('upload_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), client_id, filename, weight, round_num])

def plot_metrics_local(round_history):
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    matplotlib.use('Agg')
    sns.set_theme(style="darkgrid")

    if len(round_history) == 0:
        return

    rounds = [x["round"] for x in round_history]

    # loss
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [x.get("loss", x.get("total_loss", 0.0)) for x in round_history], marker='o', label='Loss')
    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.title("Loss vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_vs_round.png")
    plt.close()

    # accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [x.get("accuracy", x.get("anomaly_accuracy", 0.0)) for x in round_history], marker='o', label='Accuracy')
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_round.png")
    plt.close()

    # F1 & ROC AUC
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [x.get("f1", x.get("disease_f1", 0.0)) for x in round_history], marker='o', label='F1 Score')
    plt.plot(rounds, [x.get("roc_auc", 0.5) for x in round_history], marker='s', label='ROC AUC')
    plt.xlabel("Federated Round")
    plt.ylabel("Score")
    plt.title("F1 Score & ROC AUC vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("f1_vs_round.png")
    plt.close()

    # Resources
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Federated Round')
    ax1.set_ylabel('Avg Latency (seconds)', color=color)
    ax1.plot(rounds, [x.get("avg_comp_latency", 0) for x in round_history], marker='o', color=color, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Total Round Energy (Joules)', color=color)
    ax2.plot(rounds, [x.get("total_round_energy", 0) for x in round_history], marker='s', color=color, label='Energy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("System Resource Profile vs Federated Round")
    fig.tight_layout()
    plt.grid(True, linestyle=':')
    plt.savefig("system_resources_vs_round.png")
    plt.close()
    print("Metric plots saved locally.")

def get_client_password_hash(client_id: str) -> str:
    credentials_path = os.path.join(os.path.dirname(__file__), "credentials.json")
    if os.path.exists(credentials_path):
        try:
            with open(credentials_path, "r") as f:
                creds = json.load(f)
                return creds.get(client_id)
        except Exception as e:
            print(f"Warning: Failed to load credentials.json: {e}")
    return None

@app.post("/initiate")
async def initiate_auth(payload: dict = Body(...)):
    client_id = payload.get("client_id")
    if not client_id:
        return JSONResponse(status_code=400, content={"error": "Missing client_id"})
    
    stored_hash = get_client_password_hash(client_id)
    if not stored_hash:
        return JSONResponse(status_code=401, content={"error": "Unauthorized client ID"})
    
    challenge = os.urandom(32).hex()
    await redis_async.setex(f"fl:challenge:{client_id}", 300, challenge)
    
    return {"challenge": challenge}

@app.post("/authenticate")
async def authenticate_client(payload: dict = Body(...)):
    client_id = payload.get("client_id")
    response = payload.get("response")
    specs = payload.get("specs", {})
    
    if not client_id or not response:
        return JSONResponse(status_code=400, content={"error": "Missing client_id or response"})
    
    challenge = await redis_async.get(f"fl:challenge:{client_id}")
    if not challenge:
        return JSONResponse(status_code=400, content={"error": "No active challenge for this client"})
    
    stored_hash = get_client_password_hash(client_id)
    if not stored_hash:
        return JSONResponse(status_code=401, content={"error": "Unauthorized client ID"})
    
    expected = hashlib.sha256((stored_hash + challenge).encode('utf-8')).hexdigest()
    
    if not hmac.compare_digest(response, expected):
        print(f"Invalid authentication response from client: {client_id}")
        return JSONResponse(status_code=401, content={"error": "Authentication failed"})
    
    await redis_async.delete(f"fl:challenge:{client_id}")
    await redis_async.setex(f"fl:auth:{client_id}", 86400, "true")
    
    # Register client spec mapping CPU profile
    cpu_freq = float(specs.get("cpu_frequency", 2.0e9))
    tx_power = float(specs.get("tx_power", 0.2))
    
    # Store numeric ID mapping dynamically
    num_id = await redis_async.hget("fl:client_id_map", client_id)
    if num_id is None:
        num_id = await redis_async.incr("fl:next_numeric_id") - 1
        await redis_async.hset("fl:client_id_map", client_id, str(num_id))
    else:
        num_id = int(num_id)

    # Persist connection profile in Redis HASH
    profile = {
        "cpu_frequency": cpu_freq,
        "tx_power": tx_power,
        "r_trans": 15e6
    }
    await redis_async.hset("fl:client_profiles", str(num_id), json.dumps(profile))
    
    print(f"Client {client_id} successfully authenticated and registered (numeric ID: {num_id}).")
    return {"status": "authenticated", "your_id": client_id}


class FederatedServer:
    def __init__(self, selector=None, aggregator=None):
        self.is_running = False
        
        # Build profiles placeholder (will sync with Redis values at runtime)
        profiles = {}
        for i in range(100):
            profiles[i] = {
                "cpu_frequency": 2.0e9,
                "tx_power": 0.2,
                "r_trans": 15e6
            }
            
        self.env = FederatedEnv(profiles, model_size_bits=10_000_000)
        self.agent = QLearningAgent()
        
        # Load Q-table from Redis if available
        try:
            q_data = redis_sync.get("fl:rl:q_table")
            if q_data:
                q_dict = json.loads(q_data)
                self.agent.q_table = {k: np.array(v, dtype=np.float32) for k, v in q_dict.items()}
                print("Successfully loaded RL agent Q-table from Redis.")
        except Exception as e:
            print(f"Could not load Q-table: {e}")
            
        self.selector = selector or RLClientSelector(self.agent, self.env)
        self.aggregator = aggregator or FedAvg()

    async def start(self):
        global K, ROUNDS
        
        rounds_left = ROUNDS
        current_round = 0
        bucket = get_bucket_name()
        
        await redis_async.set("fl:current_round", "0")
        await redis_async.set("fl:rounds_left", str(rounds_left))
        
        round_history = []
        
        while rounds_left > 0:
            print(f"Starting round {current_round + 1}, rounds left: {rounds_left}")
            
            # Reset responses and uploads list for the round
            await redis_async.delete(f"fl:round:{current_round}:uploads")
            
            active_clients = list(await redis_async.smembers("fl:active_clients"))
            if len(active_clients) == 0:
                print("No clients connected. Pausing orchestrator...")
                await asyncio.sleep(5)
                continue
                
            # Load current client profiles from Redis to keep env updated
            client_id_map_raw = await redis_async.hgetall("fl:client_id_map")
            client_id_map = {k: int(v) for k, v in client_id_map_raw.items()}
            
            profiles_raw = await redis_async.hgetall("fl:client_profiles")
            for num_id_str, p_json in profiles_raw.items():
                self.env.profiles[int(num_id_str)] = json.loads(p_json)
                
            client_samples_raw = await redis_async.hgetall("fl:client_samples")
            client_samples = {k: float(v) for k, v in client_samples_raw.items()}
            
            client_losses_raw = await redis_async.hgetall("fl:client_losses")
            client_losses = {k: float(v) for k, v in client_losses_raw.items()}
            
            context = {
                "round": current_round + 1,
                "rounds_left": rounds_left,
                "env": self.env,
                "client_id_map": client_id_map,
                "client_samples": client_samples,
                "client_losses": client_losses
            }
            
            # Run Selector
            selected_ids = self.selector.select_clients(active_clients, K, context=context)
            selected_set = set(selected_ids)
            print(f"Selected clients for training: {selected_ids}")
            
            # Generate GET presigned URL for current global model weights
            global_model_key = f"models/global/global_model_{current_round}.keras"
            download_url = generate_presigned_download_url(global_model_key)
            
            mode_command = "train_fv" if self.aggregator.mode == "gradients" else "train"
            
            # Broadcast commands via Redis Pub/Sub channels
            for cid in active_clients:
                if cid in selected_set:
                    # Generate presigned PUT URL for local weights upload
                    cid_name = cid if cid.startswith("client_") else f"client_{cid}"
                    if mode_command == "train_fv":
                        client_upload_key = f"models/round_{current_round + 1}/{cid_name}_gradients.npz"
                    else:
                        client_upload_key = f"models/round_{current_round + 1}/{cid_name}.keras"
                        
                    upload_url = generate_presigned_upload_url(client_upload_key)
                    payload = {
                        "command": mode_command,
                        "download_url": download_url,
                        "upload_url": upload_url
                    }
                else:
                    payload = {"command": "wait"}
                    
                await redis_async.publish(f"client:ws:{cid}", json.dumps(payload))
                if cid in selected_set:
                    await redis_async.hset("fl:client_start_time", cid, str(time.time()))
                
            # Wait for selected client uploads with timeout + disconnect recovery
            start_wait = time.time()
            try:
                while True:
                    uploads_raw = await redis_async.lrange(f"fl:round:{current_round}:uploads", 0, -1)
                    uploads = [json.loads(u) for u in uploads_raw]
                    uploaded_ids = {u["client_id"] for u in uploads}
                    
                    if len(uploaded_ids) >= len(selected_ids):
                        print('all clients uploaded')
                        break
                        
                    elapsed = time.time() - start_wait
                    current_active = await redis_async.smembers("fl:active_clients")
                    disconnected = [cid for cid in selected_ids if cid not in current_active]
                    
                    if disconnected:
                        print(f"Selected client(s) {disconnected} disconnected! Re-running selection...")
                        raise RuntimeError("Client disconnected")
                        
                    if elapsed > 300:
                        print("Training phase timed out. Re-running selection...")
                        raise RuntimeError("Round timed out")
                        
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"Round training phase interrupted: {e}. Retrying selection in 5s...")
                await asyncio.sleep(5)
                continue
                
            elapsed_round = time.time() - start_wait
            print(f"Round training phase completed in {elapsed_round:.4f}s.")
            
            # Prepare configuration config dictionary to pass to Celery
            strategy_config = {}
            if isinstance(self.aggregator, qFedAvg):
                strategy_config["q"] = self.aggregator.q
            elif isinstance(self.aggregator, FedAdam):
                strategy_config["lr"] = self.aggregator.lr
                strategy_config["beta1"] = self.aggregator.beta1
                strategy_config["beta2"] = self.aggregator.beta2
                strategy_config["epsilon"] = self.aggregator.epsilon
            elif isinstance(self.aggregator, FedFV):
                strategy_config["num_clients"] = self.aggregator.num_clients
                strategy_config["alpha"] = self.aggregator.alpha
                strategy_config["tau"] = self.aggregator.tau
                
            # Map client uploads for Celery payload
            uploads_payload = []
            for u in uploads:
                cid = u["client_id"]
                uploads_payload.append({
                    "client_id": cid,
                    "numeric_id": client_id_map.get(cid, 0),
                    "s3_key": u["s3_key"],
                    "samples": u["samples"],
                    "loss": u["loss"],
                    "comp_latency": u["comp_latency"],
                    "measured_energy": u["measured_energy"],
                    "download_latency": u.get("download_latency", 0.0)
                })
                
            print(f"[DEBUG] Dispatching Celery aggregation task: strategy={self.aggregator.__class__.__name__}, round={current_round + 1}, uploads_count={len(uploads_payload)}, bucket={bucket}")
            task = aggregate_models_task.delay(
                strategy_name=self.aggregator.__class__.__name__,
                strategy_config=strategy_config,
                client_uploads=uploads_payload,
                current_round=current_round + 1,
                s3_bucket=bucket
            )
            
            # Poll Celery status
            while not task.ready():
                print(f"[DEBUG] Polling Celery task {task.id} (current state: {task.state})")
                await asyncio.sleep(0.5)
                
            celery_res = task.result
            print(f"[DEBUG] Celery task finished. Result type: {type(celery_res)}, state: {task.state}")
            
            if task.failed() or isinstance(celery_res, Exception) or not isinstance(celery_res, dict):
                print(f"[ERROR] Celery aggregation task failed: {celery_res}")
                if hasattr(task, "traceback") and task.traceback:
                    print(f"[ERROR] Celery task traceback:\n{task.traceback}")
                # Reset coordinator state and exit
                await redis_async.set("fl:is_running", "false")
                self.is_running = False
                return
                
            if celery_res.get("status") != "success":
                print(f"[ERROR] Celery aggregation returned unsuccessful status: {celery_res}")
                await redis_async.set("fl:is_running", "false")
                self.is_running = False
                return
                
            # If gradients-based, broadcast final aggregated gradients back to clients
            if self.aggregator.mode == "gradients":
                global_gt = celery_res["global_gradients"]
                for cid in selected_ids:
                    payload = {
                        "command": "apply_gradients",
                        "global_gradients": global_gt
                    }
                    await redis_async.publish(f"client:ws:{cid}", json.dumps(payload))
                    
            # Increment round numbers
            current_round += 1
            rounds_left -= 1
            await redis_async.set("fl:current_round", str(current_round))
            await redis_async.set("fl:rounds_left", str(rounds_left))
            
            # Log S3 weights/gradients uploads locally and cache values in Redis
            for u in uploads:
                cid = u["client_id"]
                log_upload(cid, u["s3_key"], u["samples"], current_round)
                await redis_async.hset("fl:client_samples", cid, str(u["samples"]))
                await redis_async.hset("fl:client_losses", cid, str(u["loss"]))
                await redis_async.hset("fl:client_latency", cid, str(u["comp_latency"]))
                await redis_async.hset("fl:client_energy", cid, str(u["measured_energy"]))
                
            # 4. Standard Evaluation Flow (Kept Uniform)
            print("Starting evaluation phase...")
            await redis_async.delete(f"fl:round:{current_round}:evals")
            
            new_global_key = f"models/global/global_model_{current_round}.keras"
            eval_download_url = generate_presigned_download_url(new_global_key)
            
            # Broadcast eval commands
            all_active = list(await redis_async.smembers("fl:active_clients"))
            for cid in all_active:
                payload = {
                    "command": "eval",
                    "download_url": eval_download_url
                }
                await redis_async.publish(f"client:ws:{cid}", json.dumps(payload))
                
            # Gather evaluation results
            start_eval = time.time()
            while True:
                evals_raw = await redis_async.lrange(f"fl:round:{current_round}:evals", 0, -1)
                evals = [json.loads(ev) for ev in evals_raw]
                evaluated_ids = {ev["client_id"] for ev in evals}
                
                current_active = await redis_async.smembers("fl:active_clients")
                if len(evaluated_ids) >= len(current_active):
                    break
                    
                elapsed_eval = time.time() - start_eval
                if elapsed_eval > 120:
                    print("Evaluation phase timed out. Processing partial metrics...")
                    break
                await asyncio.sleep(1)
                
            if evals:
                total_samples = sum(ev["samples"] for ev in evals)
                metric_names = evals[0]["metrics"].keys()
                round_metrics = {}
                for metric in metric_names:
                    weighted_metric = 0.0
                    for ev in evals:
                        weighted_metric += (ev["metrics"][metric] * (ev["samples"] / total_samples))
                    round_metrics[metric] = weighted_metric
                round_metrics["round"] = current_round
                
                # Fetch system metrics
                total_latencies = [u["comp_latency"] for u in uploads]
                total_energies = [u["measured_energy"] for u in uploads]
                total_dl_latencies = [u.get("download_latency", 0.0) for u in uploads]
                
                round_metrics["avg_comp_latency"] = np.mean(total_latencies) if total_latencies else 0.0
                round_metrics["max_comp_latency"] = np.max(total_latencies) if total_latencies else 0.0
                round_metrics["avg_energy_consumed"] = np.mean(total_energies) if total_energies else 0.0
                round_metrics["total_round_energy"] = np.sum(total_energies) if total_energies else 0.0
                round_metrics["avg_download_latency"] = np.mean(total_dl_latencies) if total_dl_latencies else 0.0
                
                round_history.append(round_metrics)
                await redis_async.set("fl:round_history", json.dumps(round_history))
                
                with open("global_metrics.txt", "a") as f:
                    f.write(str(round_metrics) + "\n")
                    
                # Plot and upload to S3
                plot_metrics_local(round_history)
                for plot_file in ["loss_vs_round.png", "accuracy_vs_round.png", "f1_vs_round.png", "system_resources_vs_round.png"]:
                    if os.path.exists(plot_file):
                        upload_file(plot_file, f"plots/{plot_file}")
                        
            # --- Update Client Selector Policy ---
            if round_history:
                current_loss = round_history[-1].get("loss", round_history[-1].get("total_loss", 0.0))
                prev_loss = round_history[-2].get("loss", round_history[-2].get("total_loss", current_loss)) if len(round_history) > 1 else current_loss
                global_loss_delta = prev_loss - current_loss
                
                round_summary = {
                    "round": current_round,
                    "rounds_left": rounds_left,
                    "selected_ids": selected_ids,
                    "active_clients": active_clients,
                    "client_id_map": client_id_map,
                    "client_samples": client_samples,
                    "client_losses": client_losses,
                    "global_loss_delta": global_loss_delta,
                    "local_losses": [client_losses.get(cid, 1.0) for cid in selected_ids],
                    "elapsed_round": elapsed_round,
                    "client_roundtrips": {
                        cid: float(await redis_async.hget("fl:client_roundtrip", cid) or elapsed_round)
                        for cid in selected_ids
                    },
                    "client_latencies": {
                        cid: float(await redis_async.hget("fl:client_latency", cid) or 1.0)
                        for cid in selected_ids
                    },
                    "client_energies": {
                        cid: float(await redis_async.hget("fl:client_energy", cid) or 5.0)
                        for cid in selected_ids
                    }
                }
                
                # Polymorphic policy update for any selector strategy (Q-Learning, PPO, DQN, AlphaZero, etc.)
                self.selector.update_policy(round_summary)
                
                # Persist Q-table to Redis if agent maintains a q_table
                if hasattr(self.selector, "agent") and hasattr(self.selector.agent, "q_table"):
                    await redis_async.set("fl:rl:q_table", json.dumps({k: v.tolist() for k, v in self.selector.agent.q_table.items()}))
                    
                # Broadcast latest metrics to active clients
                metrics_payload = json.dumps(round_history[-1])
                for cid in all_active:
                    await redis_async.publish(f"client:ws:{cid}", json.dumps({
                        "command": "metrics",
                        "payload": metrics_payload
                    }))

        print("Federated learning rounds completed.")
        # Exit instruction
        all_active = list(await redis_async.smembers("fl:active_clients"))
        for cid in all_active:
            await redis_async.publish(f"client:ws:{cid}", json.dumps({"command": "exit"}))
            
        await redis_async.set("fl:is_running", "false")
        self.is_running = False


# Create singleton manager
manager = FederatedServer()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # Verify auth session
    is_authenticated = await redis_async.exists(f"fl:auth:{client_id}")
    if not is_authenticated:
        await websocket.close(code=1008)
        return
        
    await websocket.accept()
    await redis_async.sadd("fl:active_clients", client_id)
    
    try:
        # Client sends "ready" handshake
        msg = await websocket.receive_text()
        if msg != "ready":
            print(f"Unexpected handshake from {client_id}: {msg}")
            await websocket.close()
            return
    except Exception as e:
        print(f"Error checking handshake for {client_id}: {e}")
        await redis_async.srem("fl:active_clients", client_id)
        return
        
    # Bridge Redis Pub/Sub to Client WebSocket
    pubsub = redis_async.pubsub()
    await pubsub.subscribe(f"client:ws:{client_id}")

    # Check client start count
    active_count = await redis_async.scard("fl:active_clients")
    is_running = await redis_async.get("fl:is_running") == "true"
    
    if active_count >= N and not is_running:
        # Acquire coordinator lock
        lock = redis_async.lock("fl:coordinator_lock", timeout=3600)
        if await lock.acquire(blocking=False):
            await redis_async.set("fl:is_running", "true")
            manager.is_running = True
            print("Required client count met. Launching FL Coordinator task...")
            asyncio.create_task(manager.start())
    
    async def redis_to_ws():
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await websocket.send_text(message["data"])
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in Redis-to-WS bridge for client {client_id}: {e}")
            
    async def ws_to_redis():
        try:
            while True:
                msg_text = await websocket.receive_text()
                data = json.loads(msg_text)
                
                status = data.get("status")
                current_round_val = int(await redis_async.get("fl:current_round") or "0")
                
                if status == "done":
                    # Client uploaded trained weights/gradients to S3
                    # Append result parameters to Redis List
                    await redis_async.rpush(f"fl:round:{current_round_val}:uploads", json.dumps({
                        "client_id": client_id,
                        "s3_key": data["s3_key"],
                        "samples": float(data["samples"]),
                        "loss": float(data["loss"]),
                        "comp_latency": float(data["comp_latency"]),
                        "measured_energy": float(data["measured_energy"]),
                        "download_latency": float(data["download_latency"])
                    }))
                    print(f"[WebSocket Bridge] Upload registration logged for client {client_id}.")
                    
                    # Calculate and save individual round-trip latency
                    start_time_raw = await redis_async.hget("fl:client_start_time", client_id)
                    if start_time_raw:
                        individual_rt = time.time() - float(start_time_raw)
                        await redis_async.hset("fl:client_roundtrip", client_id, str(individual_rt))
                        print(f"[WebSocket Bridge] Individual roundtrip for {client_id}: {individual_rt:.4f}s")
                    
                elif status == "evaluated":
                    # Client completed local evaluation metrics
                    await redis_async.rpush(f"fl:round:{current_round_val}:evals", json.dumps({
                        "client_id": client_id,
                        "samples": float(data["samples"]),
                        "metrics": data["metrics"]
                    }))
                    print(f"[WebSocket Bridge] Evaluation registration logged for client {client_id}.")
                    
        except WebSocketDisconnect:
            print(f"Client #{client_id} connection disconnected.")
        except Exception as e:
            print(f"Error in WS-to-Redis loop for client {client_id}: {e}")
        finally:
            await redis_async.srem("fl:active_clients", client_id)
            await pubsub.unsubscribe(f"client:ws:{client_id}")
            
    # Launch concurrent WS loops
    ws_task = asyncio.create_task(ws_to_redis())
    redis_task = asyncio.create_task(redis_to_ws())
    
    try:
        await asyncio.gather(ws_task, redis_task)
    except Exception:
        pass
    finally:
        ws_task.cancel()
        redis_task.cancel()
        await redis_async.srem("fl:active_clients", client_id)

# ----------------------------------------------------
# S3 MOCK ENDPOINTS (For offline local verification)
# ----------------------------------------------------
from fastapi.responses import FileResponse
from fastapi import Request

MOCK_S3_DIR = os.environ.get("MOCK_S3_DIR", "tmp_s3_bucket")

@app.get("/mock-s3/download")
async def mock_s3_download(key: str):
    file_path = os.path.join(MOCK_S3_DIR, key)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": f"File not found in mock S3: {key}"})
    return FileResponse(file_path)

@app.put("/mock-s3/upload")
async def mock_s3_upload(key: str, request: Request):
    file_path = os.path.join(MOCK_S3_DIR, key)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    body = await request.body()
    with open(file_path, "wb") as f:
        f.write(body)
    print(f"[Mock S3 Route] Saved uploaded file to {file_path} (size: {len(body)} bytes)")
    return {"status": "success"}