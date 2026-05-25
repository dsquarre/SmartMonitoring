import os
import sys
import io
import csv
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form, Body,HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tensorflow as tf
from model import Model
from collections import deque
import random
import numpy as np
import bcrypt

#vars
rounds_left = 10
clients = set()
model = Model()
app = FastAPI()
current_round = 0
next_round = 1
client_queues = deque()
client_metrics = deque()
done = False
global_metrics = {}
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
    a = 'abcdefghijklmnopqrstuvwxyz!@#$%&ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
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
        writer.writerow([datetime.now().isoformat(), client_id, filename, weight, next_round])

def aggregate(client_data,global_model_path):
    global_model = Model()
    total_samples = sum(samples for _, samples in client_data)

    aggregated_weights = None
    for model_path, client_samples in client_data:

        local_model = Model()
        local_model.model.load_weights(model_path)

        # [weights, biases, weights, biases, ...]
        local_weights = local_model.model.get_weights()

        if aggregated_weights is None:

            aggregated_weights = [

                np.zeros_like(layer)

                for layer in local_weights
            ]

        for i in range(len(local_weights)):

            aggregated_weights[i] += (

                local_weights[i]

                * (client_samples / total_samples)

            )

    global_model.model.set_weights(aggregated_weights)
    global_model.model.save(global_model_path)


def aggregate_models():
    global client_queues,current_round, next_round,rounds_left,client_metrics
    client_metrics.clear()
    if len(client_queues) > 2:
        print('starting aggregation')
        current_round += 1
        client_data = []
        files_to_delete = []
        for i in range(len(client_queues)):
            file_info,samples = client_queues.popleft()
            client_data.append((file_info, samples))
            files_to_delete.append(file_info)
        files_to_delete.append(f"models/global_model_{next_round - 1}.keras")  
        aggregate(client_data, f"models/global_model_{next_round}.keras")
        for path in files_to_delete:
            if os.path.exists(path):
                os.remove(path)
        client_queues.clear()
        print(f"Round {next_round} complete.")
        rounds_left -= 1
        next_round+=1

def evaluate():
    global client_metrics,done,global_metrics,current_round,round_history
    #print(client_metrics)
    #client_metrics is a deque of tuple (local_metrics,samples)->(dict,int)
    if len(client_metrics)>2:
        total_samples = sum(samples for _, samples in client_metrics)
        metric_names = client_metrics[0][0].keys()
        round_metrics = {}
        for metric in metric_names:
            weighted_metric = 0.0
            for metrics, samples in client_metrics:
                weighted_metric += (metrics[metric]* (samples/total_samples))
            round_metrics[metric] = weighted_metric
        round_metrics["round"] = current_round
        global_metrics = round_metrics
        round_history.append(round_metrics)
        with open("global_metrics.txt", "a") as f:
            f.write(str(round_metrics) + "\n")

        plot_metrics()
        client_metrics.clear()
        done = True

def plot_metrics():

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
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
    
@app.get("/evaluate")
async def get_global_eval():
    global global_metrics
    return {"global_metrics": global_metrics}

@app.get("/version")
async def version():
    global next_round, rounds_left
    print(f"Global round: {next_round}, Rounds left: {rounds_left}")
    return {"global_round": next_round, "rounds_left": rounds_left}

@app.get("/done")
async def finished():
    global done
    print(done)
    return {"message":done}

@app.get("/download")
async def get_global_model():
    global next_round
    model_path = f"models/global_model_{next_round - 1}.keras"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Global model for round {next_round - 1} not found.")
    
    return FileResponse(
        path=model_path, 
        filename=model_path,
        headers={"Global-Round": str(next_round - 1)}
    )

@app.post("/")
async def root(psswd: str = Body(...),):
    if not psswd or not authenticate(psswd):
        print('invalid password attempt')
        return {"message": "Welcome to the Federated Learning Server."}
    global done
    id = generate_id()
    clients.add(id)
    print(f"Client {id} connected. Total clients: {len(clients)}")
    return {"your_id": id}



@app.post("/upload")
async def upload_local_model(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...), 
    client_round:int = Body(...),
    samples: float = Form(1.0),
    file: UploadFile = File(...),
):
    global next_round,client_queues,clients,current_round
    if client_round<current_round:
        print('stale update dropped')
        return {"message":"stale update, file dropped"}
    if client_id not in clients:
        print('client id not in the authenticated clients')
        return {"error": "Invalid client ID. Please connect to the server first to get a valid ID."}

    file_path = f"models/client_{client_id}_model_{next_round}.keras"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    client_queues.append((file_path, samples))
    log_upload(client_id, file_path, samples)
    
    print(f"Received update from {client_id}. clients in queue: {len(client_queues)}")
    background_tasks.add_task(aggregate_models)
    return {"message": "uploaded successfully"}

@app.post("/eval_upload")
async def eval_upload(
    background_tasks: BackgroundTasks,
    client_id:str = Body(...),
    samples:float = Body(...),
    local_metrics: dict = Body(...),
):
    global clients,rounds_left,client_metrics
    if client_id not in clients:
        return {"error": "Invalid client ID. Please connect to the server first to get a valid ID."}
    done=False
    client_metrics.append((local_metrics,samples))
    log_upload(client_id, "local_metrics", samples)
    print(f"Received eval update from {client_id}. clients in queue: {len(client_metrics)}")

    background_tasks.add_task(evaluate)
    return {"message": "uploaded successfully"}