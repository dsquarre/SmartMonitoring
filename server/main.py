import os
import csv
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
import tensorflow as tf
from model import Model
from collections import deque
import random
import numpy as np
import bcrypt
import shutil
import time
from selector import RandomClientSelector, RLClientSelector, RandomRLAgent, QLearningAgent
from aggregator import FedAvg, FedFV, qFedAvg, FedAdam
from rl_env import FederatedEnv


client_id_map = {}
next_numeric_id = 0

#vars
N = 10
K = 10
rounds_left = 10
clients = set()
app = FastAPI()
current_round = 0
client_metrics = deque()
round_history = []

os.makedirs('models', exist_ok=True)

if not os.path.exists('upload_log.csv'):
    with open('upload_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "client_id", "filename", "weight", "round"])

if not os.path.exists("models/global_model_0.keras"):
    model = Model()
    model.model.save("models/global_model_0.keras")
    print(f"Created initial global model at global_model_0.keras")

def generate_id():
    a = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    n = len(a)
    l = (random.randint(16,32))
    word = ''
    for i in range(l):
        b = (random.randint(0,n-1))
        word += a[b]
    return word

def log_upload(client_id, filename, weight):
    with open('upload_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), client_id, filename, weight, current_round + 1])


def evaluate():
    global client_metrics, current_round, round_history, N
    #print(client_metrics)
    #client_metrics is a deque of tuple (local_metrics,samples)->(dict,int)
    if len(client_metrics) >= N:
        total_samples = sum(samples for _, samples in client_metrics)
        metric_names = client_metrics[0][0].keys()
        round_metrics = {}
        for metric in metric_names:
            weighted_metric = 0.0
            for metrics, samples in client_metrics:
                weighted_metric += (metrics[metric]* (samples/total_samples))
            round_metrics[metric] = weighted_metric
        round_metrics["round"] = current_round
        round_history.append(round_metrics)
        with open("global_metrics.txt", "a") as f:
            f.write(str(round_metrics) + "\n")

        plot_metrics()
        client_metrics.clear()

def plot_metrics():

    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    matplotlib.use('Agg')
    sns.set_theme(style="darkgrid")

    if len(round_history) == 0:
        return

    rounds = [
        x["round"]
        for x in round_history
    ]

    #loss
    plt.figure(figsize=(8, 5))
    plt.plot(
    rounds,
    [x["total_loss"] for x in round_history],
    marker='o',
    label='Total Loss'
    )
    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.title("Loss vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_vs_round.png")
    plt.close()

    #accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        [x["anomaly_accuracy"] for x in round_history],
        marker='o',
        label='Anomaly Accuracy'
    )
    plt.plot(
        rounds,
        [x["disease_accuracy"] for x in round_history],
        marker='o',
        label='Disease Accuracy'
    )
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_round.png")
    plt.close()

    #F1
    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        [x["disease_f1"] for x in round_history],
        marker='o'
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Disease F1 Score")
    plt.title("Disease F1 vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("f1_vs_round.png")
    plt.close()

    print("Metric plots saved.")

"""
client_metrics format:

[
    (
        {
            "anomaly_accuracy": 0.94,
            "disease_accuracy": 0.91,
            "disease_f1": 0.90
        },

        n1
    ),

    (
        {
            "anomaly_accuracy": 0.95,
            "disease_accuracy": 0.93,
            "disease_f1": 0.92
        },

        n2
    )
]
"""
def authenticate(password):
    with open("ps.dat", "rb") as f:
        stored_hash = f.read()
    if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return True
    else:
        return False

@app.post("/")
async def root(payload: dict = Body(...)):
    global next_numeric_id, client_id_map, clients
    psswd = payload.get("password")
    specs = payload.get("specs", {})
    
    if not psswd or not authenticate(psswd):
        print('invalid password attempt')
        return {"message": "Welcome to the Federated Learning Server."}
        
    id = generate_id()
    clients.add(id)
    
    if id not in client_id_map:
        num_id = next_numeric_id
        client_id_map[id] = num_id
        next_numeric_id += 1
        
        # Save client specs to profile database dynamically!
        manager.env.profiles[num_id] = {
            "cpu_frequency": float(specs.get("cpu_frequency", 2.0e9)),
            "tx_power": float(specs.get("tx_power", 0.2)),
            "r_trans": 15e6  # Default baseline upload bandwidth: 15 Mbps
        }
    
    print(f"Client {id} registered with specs: {specs}. Connected: {len(clients)}")
    return {"your_id": id}


class FederatedServer:
    def __init__(self, selector=None, aggregator=None):
        self.clients = {}
        self.client_uploads = []
        self.is_running = False
        
        # Initialize BEFL Environment and Client Profiles
        profiles = {
            i: {
                "cpu_frequency": float(random.choice([1.2e9, 1.6e9, 2.0e9, 2.4e9, 2.8e9])),
                "tx_power": float(random.choice([0.1, 0.2, 0.3, 0.4, 0.5])),
                "r_trans": float(random.choice([5e6, 10e6, 15e6, 20e6, 30e6]))
            }
            for i in range(100)
        }
        self.env = FederatedEnv(profiles, model_size_bits=10_000_000)
        self.agent = QLearningAgent()
        
        # DB to track latest client metrics persistently
        self.client_samples_db = {}
        self.client_losses_db = {}
        
        self.selector = selector or RLClientSelector(self.agent, self.env)
        self.aggregator = aggregator or FedAvg() or FedFV(num_clients=K, alpha=0.1, tau=1)

    async def connect(self, client_id, websocket: WebSocket):
        self.clients[client_id] = websocket
        msg = await websocket.receive_text()
        if msg != "ready":
            print(f"Unexpected message from {client_id}: {msg}")


    def disconnect(self,client_id):
        if client_id in self.clients:
            del self.clients[client_id]

    async def start(self):
        global rounds_left, current_round, K
        while rounds_left > 0:
            print(f"Starting round {current_round + 1}, rounds left: {rounds_left}")
            self.client_uploads = []

            client_ids = list(self.clients.keys())
            
            # Pass profiles and dynamic metrics database in context
            context = {
                "round": current_round + 1,
                "rounds_left": rounds_left,
                "env": self.env,
                "client_id_map": client_id_map,
                "client_samples": self.client_samples_db,
                "client_losses": self.client_losses_db
            }
            selected_ids = self.selector.select_clients(client_ids, K, context=context)
            selected_set = set(selected_ids)
            print(f"Selected clients for training: {selected_ids}")

            #added
            command = "train_fv" if self.aggregator.mode == "gradients" else "train"

            send_tasks = []
            for client_id, ws in self.clients.items():
                if client_id in selected_set:
                    send_tasks.append(ws.send_text(command))
                else:
                    send_tasks.append(ws.send_text("wait"))
            await asyncio.gather(*send_tasks)

            model_path = f"models/global_model_{current_round}.keras"
            with open(model_path, "rb") as f:
                model_bytes = f.read()

            send_model_tasks = []
            for client_id in selected_ids:
                ws = self.clients[client_id]
                send_model_tasks.append(ws.send_bytes(model_bytes))
            await asyncio.gather(*send_model_tasks)

            # Track response roundtrip times for selected clients
            roundtrips = {}

            if self.aggregator.mode == "gradients":
                # Receive pure un-Adamized gradient payloads
                start_time = time.time()
                recv_tasks = [self.receive_raw_gradients(cid, self.clients[cid]) for cid in selected_ids]
                results = await asyncio.gather(*recv_tasks)
                elapsed = time.time() - start_time
                for cid in selected_ids:
                    roundtrips[cid] = elapsed
                    
                self.client_uploads = [r for r in results if r is not None]
                
                current_round += 1
                next_global_model_path = f"models/global_model_{current_round}.keras"
                shutil.copyfile(model_path, next_global_model_path) # Prepare file target destination
                
                # Execute FedFV aggregation algorithm
                global_gt = self.aggregator.aggregate(
                    self.client_uploads, next_global_model_path, current_round, ModelClass=Model
                )
                
                # Manually process server weight optimization updates locally
                global_model = Model()
                global_model.model.load_weights(next_global_model_path)
                
                # FIX: Iterate and update trainable_variables directly instead of get_weights()
                for var, gg in zip(global_model.model.trainable_variables, global_gt):
                    # Perform element-wise subtraction directly on the tensor's underlying numpy array
                    var.assign(var.read_value() - gg)
                
                # Save the properly modified model file
                global_model.model.save(next_global_model_path)

                # Broadcast final updates back to clients to finish their loop sequence
                serialized_global_grad = json.dumps([g.tolist() for g in global_gt])
                broadcast_tasks = [self.clients[cid].send_text(serialized_global_grad) for cid in selected_ids]
                await asyncio.gather(*broadcast_tasks)
                
                print(f"Round {current_round} complete.")
                rounds_left -= 1
                self.client_uploads.clear()

            else:
                # Fall back to standard serialized weight transfers (.keras files)
                start_time = time.time()
                recv_tasks = []
                for client_id in selected_ids:
                    ws = self.clients[client_id]
                    recv_tasks.append(self.receive_model(client_id, ws))

                results = await asyncio.gather(*recv_tasks)
                elapsed = time.time() - start_time
                for cid in selected_ids:
                    roundtrips[cid] = elapsed
                    
                for res in results:
                    if res:
                        self.client_uploads.append(res)

                # Delegate directly to your original aggregation function layout
                self.agg()

            # 4. Standard Evaluation Flow (Kept Uniform)
            print("Starting evaluation for the round.")
            eval_send_tasks = []
            for client_id, ws in self.clients.items():
                eval_send_tasks.append(ws.send_text("eval"))
            await asyncio.gather(*eval_send_tasks)

            model_path = f"models/global_model_{current_round}.keras"
            with open(model_path, "rb") as f:
                model_bytes = f.read()

            eval_model_tasks = []
            for client_id, ws in self.clients.items():
                eval_model_tasks.append(ws.send_bytes(model_bytes))
            await asyncio.gather(*eval_model_tasks)

            eval_recv_tasks = []
            for client_id, ws in self.clients.items():
                eval_recv_tasks.append(self.receive_eval(client_id, ws))

            await asyncio.gather(*eval_recv_tasks)

            evaluate()

            # --- RL Training Step ---
            current_loss = round_history[-1]["total_loss"] if round_history else 1.0
            prev_loss = round_history[-2]["total_loss"] if len(round_history) > 1 else current_loss
            global_loss_delta = prev_loss - current_loss
            
            selected_metrics = {}
            for cid in selected_ids:
                num_id = client_id_map.get(cid, 0)
                samples = self.client_samples_db.get(cid, 1000)
                measured_rt = roundtrips.get(cid, None)
                selected_metrics[num_id] = self.env.compute_client_cost(num_id, samples, measured_rt)
                
            local_losses = [self.client_losses_db.get(cid, 1.0) for cid in selected_ids]
            reward = self.env.calculate_reward(selected_metrics, global_loss_delta, local_losses)
            print(f"[RL Environment] Round {current_round} Stats:")
            print(f"  - Delta Global Loss: {global_loss_delta:.4f}")
            if selected_metrics:
                print(f"  - Slowest Latency: {max(m['t_total'] for m in selected_metrics.values()):.4f} s")
                print(f"  - Total Energy: {sum(m['E_total'] for m in selected_metrics.values()):.6f} J")
            print(f"  - Calculated Reward: {reward:.4f}")

            if isinstance(self.selector, RLClientSelector):
                next_context = {
                    "round": current_round,
                    "rounds_left": rounds_left,
                    "env": self.env,
                    "client_id_map": client_id_map,
                    "client_samples": self.client_samples_db,
                    "client_losses": self.client_losses_db
                }
                next_state = self.selector._build_state(client_ids, next_context)
                self.selector.agent.update(
                    self.selector.last_state, 
                    self.selector.last_action, 
                    reward, 
                    next_state,
                    context=next_context
                )

            if round_history:
                latest_metrics = round_history[-1]
                metrics_payload = json.dumps(latest_metrics)
                metrics_send_tasks = []
                for client_id, ws in self.clients.items():
                    metrics_send_tasks.append(ws.send_text("metrics"))
                await asyncio.gather(*metrics_send_tasks)

                metrics_data_tasks = []
                for client_id, ws in self.clients.items():
                    metrics_data_tasks.append(ws.send_text(metrics_payload))
                await asyncio.gather(*metrics_data_tasks)

        print("Federated learning rounds completed.")

        exit_tasks = []
        for client_id, ws in self.clients.items():
            exit_tasks.append(ws.send_text("exit"))
        await asyncio.gather(*exit_tasks)
        self.is_running = False

    async def receive_raw_gradients(self, client_id, ws: WebSocket):
        global client_id_map
        try:
            json_data = await ws.receive_text()
            data = json.loads(json_data)
            client_grads = [np.array(g) for g in data["gradients"]]
            print(f"[SERVER] Successfully received raw gradients from client {client_id}")
            
            # Update dynamic metrics DB
            self.client_samples_db[client_id] = float(data["samples"])
            self.client_losses_db[client_id] = float(data["loss"])
            
            return (client_grads, float(data["samples"]), float(data["loss"]), client_id_map[client_id])
        except Exception as e:
            print(f"Error reading raw gradients from {client_id}: {e}")
            return None

    async def receive_model(self, client_id, ws: WebSocket):
        global current_round
        try:
            msg = await ws.receive_text()
            if msg == "FILE":
                samples_str = await ws.receive_text()
                samples = float(samples_str)

                #added
                loss_str = await ws.receive_text()
                loss = float(loss_str)
                
                # Update dynamic metrics DB
                self.client_samples_db[client_id] = samples
                self.client_losses_db[client_id] = loss
                
                file_bytes = await ws.receive_bytes()
                file_path = f"models/client_{client_id}_model_{current_round + 1}.keras"
                with open(file_path, "wb") as f:
                    f.write(file_bytes)

                done_msg = await ws.receive_text()
                if done_msg == "done":
                    log_upload(client_id, file_path, samples)
                    return (file_path, samples, loss, client_id)
        except Exception as e:
            print(f"Error receiving from {client_id}: {e}")
        return None

    async def receive_eval(self, client_id, ws: WebSocket):
        global client_metrics
        try:
            msg = await ws.receive_text()
            if msg == "EVAL":
                samples_str = await ws.receive_text()
                samples = float(samples_str)
                metrics_str = await ws.receive_text()
                local_metrics = json.loads(metrics_str)
                client_metrics.append((local_metrics, samples))
                
                # Update dynamic metrics DB
                self.client_samples_db[client_id] = samples
                self.client_losses_db[client_id] = local_metrics.get("total_loss", 1.0)
        except Exception as e:
            print(f"Error receiving eval from {client_id}: {e}")

    def agg(self):
        global rounds_left, current_round
        print('starting aggregation')
        if len(self.client_uploads) > 0:
            current_round += 1
            self.aggregator.aggregate(self.client_uploads, f"models/global_model_{current_round}.keras",current_round)
            for file_path, _, _, _ in self.client_uploads:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            print(f"Round {current_round} complete.")
            rounds_left -= 1
        self.client_uploads.clear()


manager = FederatedServer()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    global clients,manager,N
    if client_id not in clients:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    await manager.connect(client_id, websocket)
    if len(manager.clients) >= N and not manager.is_running:
        manager.is_running = True
        asyncio.create_task(manager.start())
    try:
        while True:
            await asyncio.sleep(3600)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client #{client_id} left the chat")