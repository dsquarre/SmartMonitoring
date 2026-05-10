import sys
import os
import tensorflow as tf
import requests
import time
import collections
from model import Model

server_url = "http://0.0.0.0:8000"
client_id = None
epochs = 5
model = Model()
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
    url = f"{server_url}/"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            auth_info = response.json()
            client_id = auth_info.get("your_id", None)
        else:
            print(f"Failed to authenticate: {response.status_code}")
    except Exception as e:
        print(f"Error during authentication: {e}")

def main():
    global model
    current_version, rounds_left = get_version()
    authenticate() # Get client_id
    samples = model.get_samples()
    if not download_model("global_model.h5"):
        print("Could not download initial model. Using initialized weights.")
    else:
        model.model.load_weights("global_model.h5")

    while rounds_left > 0:
        model.train(epochs)
        model.model.save("client_model.h5")
        while not upload_model("client_model.h5", samples):
            print("Upload failed. Retrying...")
            time.sleep(10)

        #wait for next version
        v,rounds_left = get_version()
        while v == current_version:
            time.sleep(10)
            v,rounds_left = get_version()
        current_version = v
        
        downloaded = False
        while not downloaded:
            downloaded = download_model("global_model.h5")
            if not downloaded:
                time.sleep(10)
        
        model.model.load_weights("global_model.h5")

    #local_met = model.evaluate()
    local_met = {"anomaly_accuracy": 0.95, "disease_accuracy": 0.90, "disease_f1": 0.88} #dummy metrics for testing
    with open("local_metrics.txt", "w") as f:
        f.write(str(local_met))
    while not eval_upload(local_met,samples):
            print("Upload failed. Retrying...")
            time.sleep(10)

    done = False
    while not done:
        response = requests.get(f"{server_url}/done")
        if response.status_code == 200:
            done = response.json().get("message", False)
        else:
            print(f"Failed to check completion status: {response.status_code}")
        time.sleep(10)
    
    global_met = global_metrics()
    with open("global_metrics.txt", "w") as f:
        f.write(str(global_met))

if __name__ == "__main__":
    main()
