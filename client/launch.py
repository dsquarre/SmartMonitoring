import os
import glob
import multiprocessing
import time
import requests

from main import Client

MAX_PARALLEL_CLIENTS = 3

ROUNDS = 5

DATASET_DIR = "../datasets/mitbih_data"

SERVER_URL = None
with open("url.txt", "r") as f:
    SERVER_URL = f.read().strip()


def get_server_round():
    try:
        response = requests.get( f"{SERVER_URL}/version")
        data = response.json()
        
        return data["available_download"]

    except Exception as e:
        print(f"Error checking server round: {e}")
        return -1

def launch():
    # collect dataset
    dataset_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.npz")))

    print("\nDatasets Found:\n")
    for file in dataset_files:
        print(file)
    total_clients = len(dataset_files)
    print(f"\nTotal Clients: {total_clients}")

    for round_num in range(ROUNDS):

        print("\n===================================")
        print(f"STARTING ROUND {round_num + 1}")
        print("===================================\n")

        active_processes = []

        for dataset_path in dataset_files:

            # limit parallel clients
            while len(active_processes) >= MAX_PARALLEL_CLIENTS:
                for p in active_processes[:]:
                    if not p.is_alive():
                        p.join()
                        active_processes.remove(p)
                time.sleep(2)

            print(f"\nLaunching client:")
            print(dataset_path)
            client = Client(dataset_path)

            p = multiprocessing.Process(target=client.start)
            p.start()
            active_processes.append(p)
            time.sleep(1)

        for p in active_processes:
            p.join()

        print(f"\nROUND {round_num + 1} CLIENTS FINISHED")

        print("\nWaiting for server aggregation...\n")
        expected_round = round_num + 1
        while True:
            server_round = get_server_round()
            print(f"Server Round: {server_round} | "f"Expected: {expected_round}")

            if server_round >= expected_round:
                print("\nAggregation complete.\n")
                break
            time.sleep(5)

    print("\nALL ROUNDS COMPLETED")

    print("\nSTARTING FINAL EVALUATION\n")

    eval_processes = []

    for dataset_path in dataset_files:
        client = Client(dataset_path)
        p = multiprocessing.Process(target=client.evaluate)
        p.start()
        eval_processes.append(p)
    for p in eval_processes:
        p.join()

    print("\nFINAL EVALUATION COMPLETE")

if __name__ == "__main__":

    multiprocessing.freeze_support()

    launch()