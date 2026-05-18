import sys
import os
import tensorflow as tf
import requests
import time
import collections
from model import Model
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", type=str, help="Path to dataset.npz")

args = None
model = None

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset is not None:
        model = Model(args.dataset)

server_url = None
with open("url.txt", "r") as f:
    server_url = f.read().strip()
client_id = None
epochs = 1


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


def upload_model(file_path, samples):
    global client_id
    url = f"{server_url}/upload"
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'client_id': client_id, 'samples': samples} 
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
            return version_info.get("available_download", 0),version_info.get("rounds_left",0)
        else:
            print(f"Failed to get version: {response.status_code}")
            return -1,-1
    except Exception as e:
        print(f"Error getting version: {e}")
        return -1,-1

def eval_upload(local_metrics,samples):
    global client_id
    url = f"{server_url}/eval_upload"
    try:
        response = requests.post(url,json={"client_id":client_id,"samples":samples,"local_metrics": local_metrics})
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

def authenticate():
    global client_id
    with open("psswd.txt", "r") as f:
        psswd = f.read().strip()
    url = f"{server_url}/"
    try:
        response = requests.get(url,data=psswd)
        if response.status_code == 200:
            auth_info = response.json()
            client_id = auth_info.get("your_id", None)
            if client_id is None:
                print("Authentication failed: No client ID received.")
                sys.exit(1)
            print(f"Authenticated successfully. Client ID: {client_id}")
        else:
            print(f"Failed to authenticate: {response.status_code}")
    except Exception as e:
        print(f"Error during authentication: {e}")


def evaluate_final_model():

    global model, client_id

    print("\nFINAL EVALUATION CLIENT\n")

    authenticate()
    samples = model.get_samples()
    global_model_path = f"final_model_{client_id}.h5"
    downloaded = False

    while not downloaded:
        downloaded = download_model(global_model_path)
        if not downloaded:
            time.sleep(5)

    model.model.load_weights(global_model_path)
    print("Final global model loaded.")

    local_met = model.evaluate()
    print(f"Local Metrics: {local_met}")
    uploaded = False
    
    while not uploaded:
        uploaded = eval_upload(local_met,samples)
        if not uploaded:
            time.sleep(5)

    print("Metrics uploaded.")

    done = False
    while not done:
        response = requests.get(f"{server_url}/done")

        if response.status_code == 200:
            done = response.json().get(
                "message",
                False
            )
        time.sleep(5)

    metrics = global_metrics()
    print("\nGLOBAL METRICS")
    print(metrics)

def main():
    global model, client_id
    print("starting client...")
    authenticate()
    current_version, rounds_left = get_version()

    print(f"Global Version: {current_version}")
    print(f"Rounds Left: {rounds_left}")

    if rounds_left <= 0:
        print("Training complete.")
        return

    samples = model.get_samples()
    global_model_path = f"global_model_{client_id}.h5"
    client_model_path = f"client_model_{client_id}.h5"

    #global model
    downloaded = False
    while not downloaded:
        downloaded = download_model(global_model_path)

        if not downloaded:
            print("Waiting for global model...")
            time.sleep(5)

    model.model.load_weights(global_model_path)
    print("Global model loaded.")

    model.train(epochs)
    model.model.save(client_model_path)

    uploaded = False
    while not uploaded:
        uploaded = upload_model(client_model_path,samples)
        if not uploaded:
            print("Upload failed.")
            time.sleep(5)

    print(f"Client {client_id} uploaded successfully.")

    import gc
    tf.keras.backend.clear_session()
    gc.collect()
    
    if os.path.exists(client_model_path):
        os.remove(client_model_path)
    if os.path.exists(global_model_path):
        os.remove(global_model_path)

    print(f"Client {client_id} finished.")

class Client:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

    def start(self):

        global model

        model = Model(self.dataset_path)

        main()
        
    def evaluate(self):

        global model

        model = Model(self.dataset_path)

        evaluate_final_model()
        
if __name__ == "__main__":
    main()