import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import requests
import time
import collections
from model import Model
import argparse
import asyncio
import json
import platform
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", type=str, help="Path to dataset.npz")
args = parser.parse_args()
training_lock = asyncio.Lock()
ip = None
with open("ip.txt", "r") as f:
    ip = f.read().strip()
server_url = f"http://{ip}"
ws_url = f"ws://{ip}/ws/"
os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
import websockets
import asyncio

tf.config.set_visible_devices([], 'GPU')

def detect_device_specs():
    cpu_freq = 2.0e9  # Fallback standard: 2.0 GHz
    system_name = platform.system()
    try:
        if system_name == "Linux":
            try:
                with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r") as f:
                    khz = float(f.read().strip())
                    cpu_freq = khz * 1e3
            except Exception:
                out = subprocess.check_output("lscpu | grep 'CPU max MHz'", shell=True).decode()
                mhz = float(out.split(":")[-1].strip())
                cpu_freq = mhz * 1e6
        elif system_name == "Windows":
            out = subprocess.check_output("wmic cpu get MaxClockSpeed", shell=True).decode()
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            if len(lines) > 1:
                mhz = float(lines[1])
                cpu_freq = mhz * 1e6
        elif system_name == "Darwin":
            out = subprocess.check_output("sysctl -n hw.cpufreq", shell=True).decode()
            cpu_freq = float(out.strip())
    except Exception as e:
        print(f"[Device Specs] Automated detection failed, utilizing default values: {e}")
    return {
        "cpu_frequency": cpu_freq,
        "tx_power": 0.2
    }

#client specific methods here
class Client:
    def __init__(self,filepath):
        self.client_id = None
        self.authenticate()
        self.current_round = -1
        self.model = Model(filepath)
        self.samples = self.model.get_samples()
        self.local_metrics_history = []
        self.global_metrics_history = []


    def authenticate(self):
        with open("psswd.txt", "r") as f:
            psswd = f.read().strip()
        url = f"{server_url}/"
        try:
            specs = detect_device_specs()
            payload = {
                "password": psswd,
                "specs": specs
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                auth_info = response.json()
                self.client_id = auth_info.get("your_id", None)
                if self.client_id is None:
                    print("Authentication failed: No client ID received.")
                    sys.exit(1)
                print(f"Authenticated successfully. Client ID: {self.client_id}")
            else:
                print(f"Failed to authenticate: {response.status_code}")
        except Exception as e:
            print(f"Error during authentication: {e}")


    def plot_metrics(self):
        import matplotlib.pyplot as plt
        import matplotlib
        import seaborn as sns
        matplotlib.use('Agg')
        sns.set_theme(style="darkgrid")
        
        if len(self.local_metrics_history) == 0:
            return

        rounds = [x["round"] for x in self.global_metrics_history]

        # Total Loss
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, [x["total_loss"] for x in self.local_metrics_history], marker='o', label='Local Loss')
        plt.plot(rounds, [x["total_loss"] for x in self.global_metrics_history], marker='x', label='Global Loss')
        plt.xlabel("Federated Round")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Federated Round - Client {self.client_id}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"metrics/loss_vs_round_client_{self.client_id}.png")
        plt.close()

        # Accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, [x["anomaly_accuracy"] for x in self.local_metrics_history], marker='o', label='Local Anomaly Accuracy')
        plt.plot(rounds, [x["disease_accuracy"] for x in self.local_metrics_history], marker='o', label='Local Disease Accuracy')
        plt.plot(rounds, [x["anomaly_accuracy"] for x in self.global_metrics_history], marker='x', label='Global Anomaly Accuracy')
        plt.plot(rounds, [x["disease_accuracy"] for x in self.global_metrics_history], marker='x', label='Global Disease Accuracy')
        plt.xlabel("Federated Round")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs Federated Round - Client {self.client_id}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"metrics/accuracy_vs_round_client_{self.client_id}.png")
        plt.close()

        # F1 Score
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, [x["disease_f1"] for x in self.local_metrics_history], marker='o', label='Local Disease F1')
        plt.plot(rounds, [x["disease_f1"] for x in self.global_metrics_history], marker='x', label='Global Disease F1')
        plt.xlabel("Federated Round")
        plt.ylabel("Disease F1 Score")
        plt.title(f"Disease F1 vs Federated Round - Client {self.client_id}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"metrics/f1_vs_round_client_{self.client_id}.png")
        plt.close()
        
        print(f"[{self.client_id}] Metric plots saved.")

#methods common to all clients here

def global_metrics():
    url = f"{server_url}/evaluate"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            global_metrics = response.json().get("global_metrics", {})
            print(f"Global evaluation metrics: {global_metrics}")
            return global_metrics
        else:
            print(f"Failed to get global evaluation: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting global evaluation: {e}")
        return None

def get_version():
    url = f"{server_url}/version"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            version_info = response.json()
            return version_info.get("global_round", 0),version_info.get("rounds_left",0)
        else:
            print(f"Failed to get version: {response.status_code}")
            return -1,-1
    except Exception as e:
        print(f"Error getting version: {e}")
        return -1,-1

def download_model(save_path):
    url = f"{server_url}/download"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Global model downloaded successfully.")
            return True
        else:
            print(f"Failed to download model: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False



async def simulate(client):
    try:
        async with websockets.connect(ws_url+client.client_id, max_size=None) as ws:
            await ws.send("ready")
            while True:
                msg = await ws.recv()
                if msg == "train":
                    print(f"[{client.client_id}] selected for training")
                    model_bytes = await ws.recv()
                    if isinstance(model_bytes, str):
                        model_bytes = model_bytes.encode('utf-8')
                    model_path = f"models/global_model_{client.client_id}.keras"
                    with open(model_path, "wb") as f:
                        f.write(model_bytes)
                    client.model.model.load_weights(model_path)
                    
                    async with training_lock:
                        #added
                        pre_train_metrics = await asyncio.to_thread(client.model.evaluate)
                        train_loss = pre_train_metrics["total_loss"]

                        await asyncio.to_thread(client.model.train, 1)
                        client_model_path = f"models/client{client.client_id}_model.keras"
                        client.model.model.save(client_model_path)
                        await ws.send("FILE")
                        await ws.send(str(client.samples))
                        await ws.send(str(train_loss))
                        with open(client_model_path, "rb") as f:
                            await ws.send(f.read())
                        await ws.send("done")
                    client.current_round += 1

                elif msg == "train_fv":
                    # --- ROUTINE B: PURE GRADIENT CONFLICT STRATEGIES (FedFV) ---
                    print(f"[{client.client_id}] Gradient FedFV execution")
                    model_bytes = await ws.recv()
                    if isinstance(model_bytes, str): model_bytes = model_bytes.encode('utf-8')
                    model_path = f"models/global_model_{client.client_id}.keras"
                    with open(model_path, "wb") as f: f.write(model_bytes)
                    client.model.model.load_weights(model_path)

                    async with training_lock:
                        # Extract un-Adamized raw structural updates using custom local loops
                        local_grads, current_loss = await asyncio.to_thread(client.model.train_local_gradients_fv)
                        
                    payload = {
                            "gradients": [g.tolist() for g in local_grads],
                            "loss": current_loss,
                            "samples": client.samples
                        }
                    await ws.send(json.dumps(payload))
                        
                        # Receive resolved steps back and apply manually
                    server_response = await ws.recv()
                    global_gradients = json.loads(server_response)
                    async with training_lock:
                        await asyncio.to_thread(client.model.apply_global_gradients_fv, global_gradients, server_lr=0.001)
                    
                    client.current_round += 1
                    
                elif msg == "eval":
                    print(f"[{client.client_id}] starting evaluation")
                    model_bytes = await ws.recv()
                    if isinstance(model_bytes, str):
                        model_bytes = model_bytes.encode('utf-8')
                    model_path = f"models/global_model_{client.client_id}.keras"
                    with open(model_path, "wb") as f:
                        f.write(model_bytes)
                    client.model.model.load_weights(model_path)
                    
                    async with training_lock:
                        local_met = await asyncio.to_thread(client.model.evaluate)
                        client.local_metrics_history.append(local_met)

                    await ws.send("EVAL")
                    await ws.send(str(client.samples))
                    await ws.send(json.dumps(local_met))
                elif msg == "metrics":
                    metrics_str = await ws.recv()
                    global_met = json.loads(metrics_str)
                    client.global_metrics_history.append(global_met)
                    print(f"[{client.client_id}] received global metrics")
                elif msg == "wait":
                    print(f"[{client.client_id}] Not selected")
                elif msg == "exit":
                    print(f"[{client.client_id}] Finished Training and Evaluation")
                    client.plot_metrics()
                    return
    except websockets.exceptions.ConnectionClosed:
        print(f"[Client {client.client_id}] Server closed the connection.")


async def main():
    clients = []
    n = 10
    for i in range(n): #no of sequential clients
        path = args.dataset + f"{i}.npz"
        client = Client(path)
        clients.append(client)
    
    tasks = [simulate(client) for client in clients]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())