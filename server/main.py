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
from selector import RandomClientSelector
from aggregator import FedAvg

#vars
N = 5
K = 3
rounds_left = 3
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
async def root(psswd: str = Body(...),):
    if not psswd or not authenticate(psswd):
        print('invalid password attempt')
        return {"message": "Welcome to the Federated Learning Server."}
    id = generate_id()
    clients.add(id)
    print(f"Client {id} connected. Total clients: {len(clients)}")
    return {"your_id": id}


class FederatedServer:
    def __init__(self, selector=None, aggregator=None):
        self.clients = {}
        self.client_uploads = []
        self.is_running = False
        self.selector = selector or RandomClientSelector()
        self.aggregator = aggregator or FedAvg()

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
            selected_ids = self.selector.select_clients(client_ids, K, context={"round": current_round + 1})
            selected_set = set(selected_ids)
            print(f"Selected clients for training: {selected_ids}")

            send_tasks = []
            for client_id, ws in self.clients.items():
                if client_id in selected_set:
                    send_tasks.append(ws.send_text("train"))
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

            recv_tasks = []
            for client_id in selected_ids:
                ws = self.clients[client_id]
                recv_tasks.append(self.receive_model(client_id, ws))

            results = await asyncio.gather(*recv_tasks)

            for res in results:
                if res:
                    self.client_uploads.append(res)

            self.agg()

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

    async def receive_model(self, client_id, ws: WebSocket):
        global current_round
        try:
            msg = await ws.receive_text()
            if msg == "FILE":
                samples_str = await ws.receive_text()
                samples = float(samples_str)
                file_bytes = await ws.receive_bytes()
                file_path = f"models/client_{client_id}_model_{current_round + 1}.keras"
                with open(file_path, "wb") as f:
                    f.write(file_bytes)

                done_msg = await ws.receive_text()
                if done_msg == "done":
                    log_upload(client_id, file_path, samples)
                    return (file_path, samples)
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
        except Exception as e:
            print(f"Error receiving eval from {client_id}: {e}")

    def agg(self):
        global rounds_left, current_round
        print('starting aggregation')
        if len(self.client_uploads) > 0:
            current_round += 1
            self.aggregator.aggregate(self.client_uploads, f"models/global_model_{current_round}.keras")
            for file_path, _ in self.client_uploads:
                if os.path.exists(file_path):
                    os.remove(file_path)
            old_model = f"models/global_model_{current_round - 1}.keras"
            if os.path.exists(old_model):
                os.remove(old_model)
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
