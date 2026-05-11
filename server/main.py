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
rounds_left = 5
clients = {}
model = Model()
app = FastAPI()
current_round = 1
client_queues = deque()
client_metrics = deque()
done = False
global_metrics = {}

os.makedirs('uploads', exist_ok=True)
if not os.path.exists('upload_log.csv'):
    with open('upload_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "client_id", "filename", "weight", "round"])

if not os.path.exists("global_model_0.h5"):
    model = Model()
    model.model.save("global_model_0.h5")
    print(f"Created initial global model at global_model_0.h5")

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
        writer.writerow([datetime.now().isoformat(), client_id, filename, weight, current_round])

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
    global client_queues, current_round,rounds_left
    if len(client_queues) > 2:
        client_data = []
        files_to_delete = []
        for i in range(3):
            file_info,samples = client_queues.popleft()
            client_data.append((file_info, samples))
            files_to_delete.append(file_info)
        files_to_delete.append(f"global_model_{current_round - 1}.h5")  
        aggregate(client_data, f"global_model_{current_round}.h5")
        for path in files_to_delete:
            if os.path.exists(path):
                os.remove(path)
        
        print(f"Round {current_round} complete.")
        current_round += 1
        rounds_left -= 1

def evaluate():
    global client_metrics,done,global_metrics
    print(client_metrics)
    #client is a deque of tuple (local_metrics,samples)->(dict,int)
    if len(client_metrics)>2:
        total_samples = sum(samples for _, samples in client_metrics)
        metric_names = client_metrics[0][0].keys()
        for metric in metric_names:
            weighted_metric = 0.0
            for metrics, samples in client_metrics:
                weighted_metric += (metrics[metric]* (samples/total_samples))
            global_metrics[metric] = weighted_metric
        done = True
        with open("global_metrics.txt", "w") as f:
            f.write(str(global_metrics))
    

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
    global current_round, rounds_left
    print(f"Current round: {current_round}, Rounds left: {rounds_left}")
    return {"available_download": current_round-1, "rounds_left": rounds_left}

@app.get("/done")
async def finished():
    global done
    print(done)
    return {"message":done}

@app.get("/download")
async def get_global_model():
    global current_round
    model_path = f"global_model_{current_round - 1}.h5"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Global model for round {current_round - 1} not found.")
    
    return FileResponse(
        path=model_path, 
        filename=model_path,
        headers={"Global-Round": str(current_round - 1)}
    )

@app.post("/")
async def root(psswd: str = Body(...),):
    if not psswd or not authenticate(psswd):
        print('invalid password attempt')
        return {"message": "Welcome to the Federated Learning Server."}
    global done
    id = generate_id()
    clients[id] = True
    print(f"Client {id} connected. Total clients: {len(clients)}")
    return {"your_id": id}



@app.post("/upload")
async def upload_local_model(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...), 
    samples: float = Form(1.0),
    file: UploadFile = File(...),
):
    global current_round,client_queues,clients
    if client_id not in clients or clients[client_id] is False:
        return {"error": "Invalid client ID. Please connect to the server first to get a valid ID."}

    filename = f"client_{client_id}_model_{current_round}.h5"
    file_path = os.path.join("uploads", filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    client_queues.append((file_path, samples))
    log_upload(client_id, filename, samples)
    
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
    global client_evals,clients,rounds_left
    if client_id not in clients or clients[client_id] is False:
        return {"error": "Invalid client ID. Please connect to the server first to get a valid ID."}
    if rounds_left > 0:
        return {"message": "Round not complete yet. Please wait for the next round to finish."}
    client_metrics.append((local_metrics,samples))
    log_upload(client_id, "local_metrics", samples)
    print(f"Received eval update from {client_id}. clients in queue: {len(client_metrics)}")

    background_tasks.add_task(evaluate)
    return {"message": "uploaded successfully"}

