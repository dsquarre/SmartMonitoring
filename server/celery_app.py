import os
import shutil
import json
import numpy as np
from celery import Celery
from s3_helper import download_file, upload_file

# Initialize Celery app
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "fl_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_transport_options={"protocol": 2},
    redis_backend_transport_options={"protocol": 2},
)

# Helper function to load gradients from .npz
def load_gradients_from_npz(npz_path: str):
    """
    Loads saved gradients from an .npz file and restores their order.
    """
    with np.load(npz_path) as data:
        # Sort keys to ensure arrays are restored in correct order (arr_0, arr_1, ...)
        keys = sorted(data.files, key=lambda x: int(x.split('_')[1]))
        return [data[key] for key in keys]

@celery_app.task(name="tasks.aggregate_models")
def aggregate_models_task(
    strategy_name: str,
    strategy_config: dict,
    client_uploads: list,
    current_round: int,
    s3_bucket: str
):
    """
    Celery task that downloads client updates from S3, instantiates the aggregator,
    runs aggregation, and saves the new global model to S3.
    
    client_uploads: List of dicts representing client response metadata:
        For weights mode:
            [{"client_id": "client_x", "s3_key": "...", "samples": 100, "loss": 0.5}, ...]
        For gradients mode (FedFV):
            [{"client_id": "client_x", "numeric_id": 1, "s3_key": "...", "samples": 100, "loss": 0.5, "comp_latency": 1.2, "measured_energy": 5.0}, ...]
    """
    # Import locally to avoid issues during Celery boot (since tensorflow is slow to load)
    from model import Model
    from aggregator import FedAvg, qFedAvg, FedAdam, FedFV
    
    # Using s3_helper functions for uploads and downloads
    local_dir = f"tmp_round_{current_round}"
    os.makedirs(local_dir, exist_ok=True)
    
    # Instantiate the aggregator
    if strategy_name == "FedAvg":
        aggregator = FedAvg()
    elif strategy_name == "qFedAvg":
        aggregator = qFedAvg(q=strategy_config.get("q", 0.5))
    elif strategy_name == "FedAdam":
        aggregator = FedAdam(
            lr=strategy_config.get("lr", 0.001),
            beta1=strategy_config.get("beta1", 0.9),
            beta2=strategy_config.get("beta2", 0.999),
            epsilon=strategy_config.get("epsilon", 1e-8)
        )
    elif strategy_name == "FedFV":
        aggregator = FedFV(
            num_clients=strategy_config.get("num_clients", 10),
            alpha=strategy_config.get("alpha", 0.1),
            tau=strategy_config.get("tau", 1)
        )
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy_name}")

    print(f"[Celery Worker] Starting aggregation for strategy: {strategy_name}, round: {current_round}")

    # Paths for global models
    prev_global_s3_key = f"models/global/global_model_{current_round - 1}.keras"
    next_global_s3_key = f"models/global/global_model_{current_round}.keras"
    
    prev_global_local_path = os.path.join(local_dir, f"global_model_{current_round - 1}.keras")
    next_global_local_path = os.path.join(local_dir, f"global_model_{current_round}.keras")

    print(f"[Celery Worker] Downloading previous global model from S3: {prev_global_s3_key}")
    download_file(prev_global_s3_key, prev_global_local_path)
    shutil.copyfile(prev_global_local_path, next_global_local_path)

    # Process client uploads
    client_data = []
    temp_files = []
    
    try:
        if strategy_name == "FedFV":
            # Gradients-based aggregation (FedFV)
            for upload in client_uploads:
                client_id = upload["client_id"]
                numeric_id = upload["numeric_id"]
                s3_key = upload["s3_key"]
                samples = upload["samples"]
                loss = upload["loss"]
                comp_latency = upload.get("comp_latency", 1.0)
                measured_energy = upload.get("measured_energy", 5.0)

                # Download gradients npz file
                local_npz_path = os.path.join(local_dir, f"client_{client_id}_gradients.npz")
                download_file(s3_key, local_npz_path)
                temp_files.append(local_npz_path)
                
                # Load client gradients from the npz file
                client_grads = load_gradients_from_npz(local_npz_path)
                
                client_data.append((
                    client_grads,
                    samples,
                    loss,
                    numeric_id,
                    comp_latency,
                    measured_energy
                ))
            
            # Execute FedFV aggregation
            print(f"[Celery Worker] Executing FedFV gradient conflict resolution...")
            global_gt = aggregator.aggregate(
                client_data, next_global_local_path, current_round, ModelClass=Model
            )
            
            # Apply global gradient updates to next global model
            global_model = Model()
            global_model.model.load_weights(next_global_local_path)
            for var, gg in zip(global_model.model.trainable_variables, global_gt):
                var.assign(var.read_value() - gg)
            global_model.model.save(next_global_local_path)
            
            # Serialize global gradients to list so coordinator can broadcast it to clients
            serialized_global_grads = [g.tolist() for g in global_gt]
            result = {"status": "success", "global_gradients": serialized_global_grads}
            
        else:
            # Weights-based aggregation (FedAvg, qFedAvg, FedAdam)
            for upload in client_uploads:
                client_id = upload["client_id"]
                s3_key = upload["s3_key"]
                samples = upload["samples"]
                loss = upload["loss"]
                comp_latency = upload.get("comp_latency", 1.0)
                measured_energy = upload.get("measured_energy", 5.0)
                client_dl_latency = upload.get("download_latency", 0.0)

                # Download local weights .keras file
                local_keras_path = os.path.join(local_dir, f"client_{client_id}_model.keras")
                download_file(s3_key, local_keras_path)
                temp_files.append(local_keras_path)
                
                client_data.append((
                    local_keras_path,
                    samples,
                    loss,
                    client_id,
                    comp_latency,
                    measured_energy,
                    client_dl_latency
                ))

            # Execute weights aggregation
            print(f"[Celery Worker] Executing {strategy_name} weight aggregation...")
            aggregator.aggregate(client_data, next_global_local_path, current_round)
            result = {"status": "success"}

        # Upload new global model to S3
        print(f"[Celery Worker] Uploading aggregated global model to S3: {next_global_s3_key}")
        upload_file(next_global_local_path, next_global_s3_key)
        
    finally:
        # Cleanup all temp files
        print(f"[Celery Worker] Cleaning up temporary files...")
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(prev_global_local_path):
            os.remove(prev_global_local_path)
        if os.path.exists(next_global_local_path):
            os.remove(next_global_local_path)
        try:
            os.rmdir(local_dir)
        except Exception:
            pass

    return result
