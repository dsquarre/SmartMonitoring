import sys
import os
import tensorflow as tf
import requests
import time
import collections
from model import Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", type=str, help="Path to dataset.npz")
args = parser.parse_args()

server_url = None
with open("url.txt", "r") as f:
    server_url = f.read().strip()

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

#client specific methods here
class Client:
    def __init__(self,filepath):
        self.client_id = None
        self.authenticate()
        self.current_round = -1
        self.model = Model(filepath)
        self.samples = self.model.get_samples()

    def upload_model(self,file_path):
        url = f"{server_url}/upload"
        print(f"client round: {self.current_round}")
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {'client_id': self.client_id,'client_round':self.current_round, 'samples': self.samples} 
                response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                print("Model uploaded successfully.")
                return True
            else:
                print(f"Failed to upload model: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error uploading model: {e}")
            return False

    def eval_upload(self,local_metrics):
        url = f"{server_url}/eval_upload"
        try:
            response = requests.post(url,json={"client_id":self.client_id,"samples":self.samples,"local_metrics": local_metrics})
            if response.status_code == 200:
                res = response.json()
                print(res.get("message",""))
                return True
            else:
                print(f"Failed to upload local_metrics: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error uploading local metrics: {e}")
            return False

    def authenticate(self):
        with open("psswd.txt", "r") as f:
            psswd = f.read().strip()
        url = f"{server_url}/"
        try:
            response = requests.post(url,data=psswd)
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


def simulate(clients):
    for i in range(len(clients)): #no of sequential clients
        client = clients[i]
        global_round, rounds_left = get_version()
        if client.current_round<0:
            client.current_round = global_round-1
        else:
            while global_round-1<client.current_round:
                print('waiting for latest model')
                time.sleep(10)
                global_round, rounds_left = get_version()
        if not download_model("models/global_model.keras"):
            print("Could not download global model, network error")
            sys.exit(1)
        else:
            client.model.model.load_weights("models/global_model.keras")

        if rounds_left > 0:
            client.model.train(epochs=1)
            client.current_round+=1
            client.model.model.save(f"models/client{i}_model.keras")
            while not client.upload_model(f"models/client{i}_model.keras"):
                print("Upload failed. Retrying...")
                time.sleep(10)
        else:
            print(f'rounds left : {rounds_left}')
            return False
    return True


def main():
    clients = []
    n = 3
    for i in range(n): #no of sequential clients
        path = args.dataset + f"{i}.npz"
        client = Client(path)
        clients.append(client)
    
    while simulate(clients):
        print("round done")
    for i in range(n):
        client = clients[i]
        local_met = client.model.evaluate()
        #dummy metrics for testing
        #local_met = {"anomaly_accuracy": 0.95, "disease_accuracy": 0.90, "disease_f1": 0.88}
        with open(f"metrics/local_metrics_client{i}.txt", "w") as f:
            f.write(str(local_met))
        while not client.eval_upload(local_met):
            print("Upload failed. Retrying...")
            time.sleep(10)

    done = False
    while not done:
        response = requests.get(f"{server_url}/done")
        if response.status_code == 200:
            done = response.json().get("message", False)
        else:
            print(f"Failed to check completion status: {response.status_code}")
        print('global metrics not available yet')
        time.sleep(10)
    
    global_met = global_metrics()
    with open("metrics/global_metrics.txt", "w") as f:
        f.write(str(global_met))

if __name__ == "__main__":
    main()
