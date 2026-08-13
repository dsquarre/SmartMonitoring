import sys
import os
import time
import requests
import asyncio
import json
import websockets
import hashlib
import hmac

# Resolve server URL and WebSocket URL
server_url = "http://127.0.0.1:8000"
ws_url = "ws://127.0.0.1:8000/ws/"
password = "P7h1!quiBO0no96"

def extract_s3_key(url: str) -> str:
    if "key=" in url:
        return url.split("key=")[1].split("&")[0]
    path = url.split("?")[0]
    parts = path.split(".amazonaws.com/")
    if len(parts) > 1:
        return parts[1]
    return path.split("/")[-1]

class MockClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.samples = 500
        self.current_round = 0
        self.authenticate()

    def authenticate(self):
        try:
            # 1. Initiate challenge
            response = requests.post(f"{server_url}/initiate", json={"client_id": self.client_id})
            challenge = response.json().get("challenge")
            
            # 2. Compute response
            hashed_pwd = hashlib.sha256(password.encode('utf-8')).hexdigest()
            response_hash = hashlib.sha256((hashed_pwd + challenge).encode('utf-8')).hexdigest()
            
            # 3. Authenticate
            auth_payload = {
                "client_id": self.client_id,
                "response": response_hash,
                "specs": {"cpu_frequency": 2.4e9, "tx_power": 0.2}
            }
            auth_res = requests.post(f"{server_url}/authenticate", json=auth_payload)
            if auth_res.status_code == 200:
                print(f"[{self.client_id}] Mock Client authenticated successfully.")
            else:
                print(f"[{self.client_id}] Authentication failed: {auth_res.text}")
                sys.exit(1)
        except Exception as e:
            print(f"[{self.client_id}] Error in mock auth: {e}")
            sys.exit(1)

async def run_mock_client(client_id):
    client = MockClient(client_id)
    try:
        async with websockets.connect(ws_url + client.client_id) as ws:
            await ws.send("ready")
            print(f"[{client_id}] Connected to server and sent ready handshake.")
            
            while True:
                msg_text = await ws.recv()
                event = json.loads(msg_text)
                command = event.get("command")
                
                if command == "train":
                    print(f"[{client_id}] Received train command.")
                    download_url = event["download_url"]
                    upload_url = event["upload_url"]
                    
                    # 1. Simulate download from S3
                    dl_start = time.perf_counter()
                    resp = requests.get(download_url)
                    dl_latency = time.perf_counter() - dl_start
                    print(f"[{client_id}] Mock-downloaded global weights (size: {len(resp.content)} bytes) in {dl_latency:.4f}s")
                    
                    # 2. Simulate training delay
                    print(f"[{client_id}] Simulating local training...")
                    await asyncio.sleep(1.0) # 1 second mock training
                    
                    # 3. Simulate S3 upload
                    dummy_model_bytes = b"mock_trained_model_weights_data_" + os.urandom(100)
                    up_start = time.perf_counter()
                    up_res = requests.put(upload_url, data=dummy_model_bytes)
                    up_latency = time.perf_counter() - up_start
                    print(f"[{client_id}] Mock-uploaded weights. HTTP Status: {up_res.status_code} in {up_latency:.4f}s")
                    
                    # 4. Notify server via WebSocket
                    payload = {
                        "status": "done",
                        "s3_key": extract_s3_key(upload_url),
                        "samples": client.samples,
                        "loss": 0.25 - (client.current_round * 0.02),
                        "comp_latency": 1.0,
                        "measured_energy": 4.5,
                        "download_latency": dl_latency
                    }
                    await ws.send(json.dumps(payload))
                    client.current_round += 1
                    
                elif command == "train_fv":
                    print(f"[{client_id}] Received train_fv command.")
                    download_url = event["download_url"]
                    upload_url = event["upload_url"]
                    
                    # 1. Simulate download
                    dl_start = time.perf_counter()
                    resp = requests.get(download_url)
                    dl_latency = time.perf_counter() - dl_start
                    
                    # 2. Simulate local gradients extraction
                    print(f"[{client_id}] Simulating local gradient computation...")
                    await asyncio.sleep(1.0)
                    
                    # 3. Create dummy gradients file and upload
                    dummy_gradient_bytes = b"mock_gradients_data"
                    up_start = time.perf_counter()
                    up_res = requests.put(upload_url, data=dummy_gradient_bytes)
                    up_latency = time.perf_counter() - up_start
                    print(f"[{client_id}] Mock-uploaded gradients. HTTP Status: {up_res.status_code}")
                    
                    # 4. Notify server via WebSocket
                    payload = {
                        "status": "done",
                        "s3_key": extract_s3_key(upload_url),
                        "samples": client.samples,
                        "loss": 0.35 - (client.current_round * 0.03),
                        "comp_latency": 1.0,
                        "measured_energy": 3.8,
                        "download_latency": dl_latency
                    }
                    await ws.send(json.dumps(payload))
                    
                    # Wait for global gradients
                    msg_grads = await ws.recv()
                    event_grads = json.loads(msg_grads)
                    if event_grads.get("command") == "apply_gradients":
                        print(f"[{client_id}] Received and applied mock global gradients from FedFV.")
                        
                    client.current_round += 1
                    
                elif command == "eval":
                    print(f"[{client_id}] Received eval command.")
                    download_url = event["download_url"]
                    
                    # Simulate download
                    requests.get(download_url)
                    
                    # Simulate evaluation
                    local_metrics = {
                        "total_loss": 0.20 - (client.current_round * 0.02),
                        "anomaly_accuracy": 0.85 + (client.current_round * 0.01),
                        "disease_accuracy": 0.82 + (client.current_round * 0.01),
                        "disease_f1": 0.80 + (client.current_round * 0.01)
                    }
                    
                    # Report metrics
                    payload = {
                        "status": "evaluated",
                        "samples": client.samples,
                        "metrics": local_metrics
                    }
                    await ws.send(json.dumps(payload))
                    print(f"[{client_id}] Reported mock evaluation metrics.")
                    
                elif command == "metrics":
                    print(f"[{client_id}] Received latest round metrics update.")
                    
                elif command == "wait":
                    print(f"[{client_id}] Received wait command.")
                    
                elif command == "exit":
                    print(f"[{client_id}] Received exit command. Verification successful!")
                    return
                    
    except Exception as e:
        print(f"[{client_id}] Connection error: {e}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python mock_client.py <client_id>")
        sys.exit(1)
    client_id = sys.argv[1]
    await run_mock_client(client_id)

if __name__ == "__main__":
    asyncio.run(main())
