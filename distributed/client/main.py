import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import requests
import time
import collections
from model import Model
from attacks import craft_malicious_weights
import argparse
import asyncio
import json
import platform
import subprocess
import websockets
import socket
import hashlib
import ssl
from codecarbon import EmissionsTracker
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def verify_ssl_fingerprint(host: str, port: int, expected_fingerprint: str):
    """
    Connects to the server over SSL and verifies that its certificate SHA-256 fingerprint
    matches expected_fingerprint to prevent Man-in-the-Middle (MitM) attacks.
    """
    clean_expected = expected_fingerprint.replace(":", "").lower()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with socket.create_connection((host, port), timeout=10) as sock:
        with ctx.wrap_socket(sock) as ssock:
            der_cert = ssock.getpeercert(binary_form=True)
            if not der_cert:
                raise ssl.SSLError("Server did not present an SSL certificate.")
            actual_hash = hashlib.sha256(der_cert).hexdigest().lower()
            if actual_hash != clean_expected:
                raise ssl.SSLError(
                    f"SSL Fingerprint Mismatch! Potential MitM Attack Detected.\n"
                    f"Expected: {clean_expected}\n"
                    f"Actual:   {actual_hash}"
                )
            print(f"[SSL Security] Server certificate SHA-256 fingerprint verified ({actual_hash[:16]}...).")
            return True

# Argument parsing and environment variable configuration
parser = argparse.ArgumentParser(description="Federated Learning Client Container")
parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the dataset .npz file")
parser.add_argument("-s", "--server-ip", type=str, default=os.environ.get("SERVER_IP"), help="IP address of the server")
parser.add_argument("-p", "--password", type=str, default=os.environ.get("PASSWORD"), help="Authentication password")
parser.add_argument("-c", "--client-id", type=str, default=os.environ.get("CLIENT_ID", "client_0"), help="Unique Client Identifier")
parser.add_argument("-f", "--fingerprint", type=str, default=os.environ.get("CERT_FINGERPRINT"), help="Expected SHA-256 SSL certificate fingerprint for pinning")
parser.add_argument("--no-verify", "--insecure", action="store_true", default=os.environ.get("NO_VERIFY", "").lower() in ("true", "1", "yes"), help="Bypass SSL certificate verification")
parser.add_argument("--attack", type=str, default=os.environ.get("FL_ATTACK", "none"), choices=["none", "standard"],
                    help="Poisoning experiments: 'standard' uploads 5x the honest update (FedStrategist); default none")
parser.add_argument("--attack-scale", type=float, default=float(os.environ.get("FL_ATTACK_SCALE", "5.0")),
                    help="Scale factor of the standard attack (default 5)")
parser.add_argument("--flip", action="store_true", default=os.environ.get("FL_ATTACK_FLIP", "").lower() in ("true", "1", "yes"),
                    help="Reverse the update direction (a real poisoning attack)")
args = parser.parse_args()

# Resolve Server IP
ip = args.server_ip
if not ip:
    if os.path.exists("ip.txt"):
        with open("ip.txt", "r") as f:
            ip = f.read().strip()
    else:
        ip = "localhost:8000"

# Resolve Server URL and WebSocket URL scheme dynamically
if "://" in ip:
    server_url = ip
    host = ip.split("://")[1]
    if ip.startswith("https://"):
        ws_url = f"wss://{host}/ws/"
    else:
        ws_url = f"ws://{host}/ws/"
else:
    server_url = f"http://{ip}"
    ws_url = f"ws://{ip}/ws/"

# Execute Certificate Pinning check if fingerprint is provided
fingerprint = args.fingerprint
if fingerprint and server_url.startswith("https://"):
    host_clean = host.split("/")[0]
    host_ip = host_clean.split(":")[0]
    port_num = int(host_clean.split(":")[1]) if ":" in host_clean else 443
    verify_ssl_fingerprint(host_ip, port_num, fingerprint)

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# Force CPU execution to keep simulation uniform and lightweight
tf.config.set_visible_devices([], 'GPU')

def trainable_mask(keras_model):
    """Per entry of get_weights(): True for trainable weights, False for BatchNorm moving statistics."""
    trainable = {id(v) for v in keras_model.trainable_weights}
    return [id(v) in trainable for v in keras_model.weights]


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


from urllib.parse import urlparse, parse_qs

def extract_s3_key(url: str) -> str:
    """
    Extracts the exact S3 object key from a presigned S3 URL or mock S3 URL.
    """
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    if "key" in query_params:
        return query_params["key"][0]
    
    if ".amazonaws.com/" in url:
        parts = url.split("?")[0].split(".amazonaws.com/")
        if len(parts) > 1:
            return parts[1]

    # Path-style presigned URL from an S3-compatible endpoint (S3_ENDPOINT_URL): /<bucket>/<key>
    if "X-Amz-Algorithm" in query_params:
        return parsed.path.lstrip("/").split("/", 1)[-1]

    return parsed.path.lstrip("/")



class Client:
    def __init__(self, filepath, client_id, password=None, no_verify=False):
        self.client_id = client_id
        self.password = password
        self.no_verify = no_verify
        self.model = Model(filepath)
        self.samples = self.model.get_samples()
        self.authenticate()
        self.current_round = -1
        self.local_metrics_history = []
        self.global_metrics_history = []

    def authenticate(self):
        psswd = self.password
        if not psswd:
            if os.path.exists("psswd.txt"):
                with open("psswd.txt", "r") as f:
                    psswd = f.read().strip()
            else:
                print("Error: Authentication password must be provided via -p/--password, PASSWORD env, or psswd.txt file.")
                sys.exit(1)

        try:
            # Step 1: Initiate Auth & Get Challenge
            initiate_url = f"{server_url}/initiate"
            payload = {"client_id": self.client_id}
            
            verify_ssl = not self.no_verify
            if not verify_ssl:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
            response = requests.post(initiate_url, json=payload, verify=verify_ssl)
            if response.status_code != 200:
                print(f"Failed to initiate authentication: {response.status_code} - {response.text}")
                sys.exit(1)
                
            challenge = response.json().get("challenge")
            if not challenge:
                print("Authentication failed: No challenge received from server.")
                sys.exit(1)
                
            # Step 2: Compute Cryptographic Response
            hashed_pwd = hashlib.sha256(psswd.encode('utf-8')).hexdigest()
            response_hash = hashlib.sha256((hashed_pwd + challenge).encode('utf-8')).hexdigest()
            
            # Step 3: Submit Response and specs
            authenticate_url = f"{server_url}/authenticate"
            specs = detect_device_specs()
            specs["samples"] = self.samples   # training-set size: Oort registers clients with reward = data size
            auth_payload = {
                "client_id": self.client_id,
                "response": response_hash,
                "specs": specs
            }
            
            auth_response = requests.post(authenticate_url, json=auth_payload, verify=verify_ssl)
            if auth_response.status_code == 200:
                print(f"Authenticated successfully. Client ID: {self.client_id}")
            else:
                print(f"Failed to authenticate: {auth_response.status_code} - {auth_response.text}")
                sys.exit(1)
        except Exception as e:
            print(f"Error during authentication: {e}")
            sys.exit(1)

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        import matplotlib
        import seaborn as sns
        matplotlib.use('Agg')
        sns.set_theme(style="darkgrid")
        
        if len(self.local_metrics_history) == 0:
            return

        rounds = [x["round"] for x in self.global_metrics_history]

        # Loss
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, [x.get("loss", x.get("total_loss", 0.0)) for x in self.local_metrics_history], marker='o', label='Local Loss')
        plt.plot(rounds, [x.get("loss", x.get("total_loss", 0.0)) for x in self.global_metrics_history], marker='x', label='Global Loss')
        plt.xlabel("Federated Round")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Federated Round - Client {self.client_id}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"metrics/loss_vs_round_client_{self.client_id}.png")
        plt.close()

        # Accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, [x.get("accuracy", x.get("anomaly_accuracy", 0.0)) for x in self.local_metrics_history], marker='o', label='Local Accuracy')
        plt.plot(rounds, [x.get("accuracy", x.get("anomaly_accuracy", 0.0)) for x in self.global_metrics_history], marker='x', label='Global Accuracy')
        plt.xlabel("Federated Round")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs Federated Round - Client {self.client_id}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"metrics/accuracy_vs_round_client_{self.client_id}.png")
        plt.close()

        # F1 Score & ROC AUC
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, [x.get("f1", x.get("disease_f1", 0.0)) for x in self.local_metrics_history], marker='o', label='Local F1')
        plt.plot(rounds, [x.get("f1", x.get("disease_f1", 0.0)) for x in self.global_metrics_history], marker='x', label='Global F1')
        plt.plot(rounds, [x.get("roc_auc", 0.5) for x in self.local_metrics_history], marker='s', label='Local ROC AUC')
        plt.plot(rounds, [x.get("roc_auc", 0.5) for x in self.global_metrics_history], marker='d', label='Global ROC AUC')
        plt.xlabel("Federated Round")
        plt.ylabel("Score")
        plt.title(f"F1 & ROC AUC vs Federated Round - Client {self.client_id}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"metrics/f1_vs_round_client_{self.client_id}.png")
        plt.close()

        # Local Confusion Matrix (Latest Round)
        if len(self.local_metrics_history) > 0:
            latest_local = self.local_metrics_history[-1]
            count_keys = ["tn", "fp", "fn", "tp"]
            if all(k in latest_local for k in count_keys):
                cm = np.array([[int(latest_local["tn"]), int(latest_local["fp"])],
                               [int(latest_local["fn"]), int(latest_local["tp"])]])
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Normal (0)', 'AFib (1)'],
                            yticklabels=['Normal (0)', 'AFib (1)'])
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(f"Client {self.client_id} Confusion Matrix")
                plt.tight_layout()
                plt.savefig(f"metrics/confusion_matrix_client_{self.client_id}.png")
                plt.close()
        
        print(f"[{self.client_id}] Metric plots saved.")


async def simulate(client):
    try:
        ssl_context = None
        if ws_url.startswith("wss://") and client.no_verify:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
        async with websockets.connect(ws_url + client.client_id, ssl=ssl_context, max_size=None) as ws:
            await ws.send("ready")
            while True:
                msg_text = await ws.recv()
                event = json.loads(msg_text)
                command = event.get("command")
                
                if command == "train":
                    print(f"[{client.client_id}] Selected for weights training.")
                    download_url = event["download_url"]
                    upload_url = event["upload_url"]

                    # 1. Download global model weights from S3
                    download_start = time.perf_counter()
                    response = requests.get(download_url)
                    response.raise_for_status()
                    download_latency = time.perf_counter() - download_start
                    
                    model_path = f"models/global_model_{client.client_id}.keras"
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                        
                    client.model.model.load_weights(model_path)
                    w_global = client.model.model.get_weights()
                    
                    # Evaluate pre-train metrics
                    pre_train_metrics = await asyncio.to_thread(client.model.evaluate)
                    train_loss = pre_train_metrics.get("loss", pre_train_metrics.get("total_loss", 0.0))

                    # Start emissions tracker
                    tracker = EmissionsTracker(
                        project_name=f"client_{client.client_id}_round",
                        save_to_file=False,
                        log_level="error"
                    )
                    await asyncio.to_thread(tracker.start)
                    start_time = time.perf_counter()

                    # Train locally with the server's settings (plain SGD, as in the TMLR supplement)
                    local_epochs = event.get("epochs", 5)
                    local_lr = event.get("lr", 0.1)
                    local_batch = event.get("batch_size", client.model.batch_size)
                    train_result = await asyncio.to_thread(client.model.train, local_epochs, local_lr, local_batch)
                    stat_utility = train_result["stat_utility"]

                    comp_latency = time.perf_counter() - start_time
                    await asyncio.to_thread(tracker.stop)
                    actual_energy_joules = tracker._total_energy.kWh * 3.6e6 if tracker._total_energy else 0.0

                    # Poisoning experiments: a malicious client turns its honest update into an attack
                    if args.attack != "none" or args.flip:
                        client.model.model.set_weights(craft_malicious_weights(
                            w_global, client.model.model.get_weights(), trainable_mask(client.model.model),
                            args.attack, scale_factor=args.attack_scale, flip=args.flip))
                        print(f"[{client.client_id}] Uploading a malicious update ({args.attack}, flip={args.flip}).")

                    # Save local trained model
                    client_model_path = f"models/client_{client.client_id}_model.keras"
                    client.model.model.save(client_model_path)
                    
                    # 2. Upload local model weights directly to S3
                    upload_start = time.perf_counter()
                    with open(client_model_path, "rb") as f:
                        upload_resp = requests.put(upload_url, data=f)
                    upload_resp.raise_for_status()
                    upload_latency = time.perf_counter() - upload_start
                    
                    print(f"Upload status: {upload_resp.status_code} | Download: {download_latency:.4f}s | Upload: {upload_latency:.4f}s")
                    
                    # Send completion confirmation to server via WebSocket
                    payload = {
                        "status": "done",
                        "s3_key": extract_s3_key(upload_url),
                        "samples": client.samples,
                        "loss": train_loss,
                        "stat_utility": stat_utility,
                        "comp_latency": comp_latency,
                        "measured_energy": actual_energy_joules,
                        "download_latency": download_latency
                    }
                    await ws.send(json.dumps(payload))
                    client.current_round += 1

                elif command == "eval":
                    print(f"[{client.client_id}] starting evaluation.")
                    download_url = event["download_url"]
                    
                    # Download global model weights from S3
                    response = requests.get(download_url)
                    response.raise_for_status()
                    model_path = f"models/global_model_{client.client_id}.keras"
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                        
                    client.model.model.load_weights(model_path)
                    
                    # Evaluate locally
                    local_met = await asyncio.to_thread(client.model.evaluate)
                    client.local_metrics_history.append(local_met)

                    # Send evaluation results to server
                    payload = {
                        "status": "evaluated",
                        "samples": client.samples,
                        "metrics": local_met
                    }
                    await ws.send(json.dumps(payload))
                    
                elif command == "metrics":
                    metrics_str = event["payload"]
                    global_met = json.loads(metrics_str)
                    client.global_metrics_history.append(global_met)
                    print(f"[{client.client_id}] received global metrics.")
                    
                elif command == "wait":
                    print(f"[{client.client_id}] Not selected.")
                    
                elif command == "exit":
                    print(f"[{client.client_id}] Finished Training and Evaluation.")
                    client.plot_metrics()
                    return
                    
    except websockets.exceptions.ConnectionClosed:
        print(f"[Client {client.client_id}] Server closed the connection.")


async def main():
    client = Client(args.dataset, args.client_id, args.password, args.no_verify)
    await simulate(client)

if __name__ == "__main__":
    asyncio.run(main())