#!/usr/bin/env python3
"""
simulate_fl.py - Standalone In-Process Research Baseline Execution Loop

Runs sequential/batched Federated Learning experiments directly on local client dataset
partitions (in data/iid or data/non_iid) without network/WebSocket/Redis/S3/Celery overhead.
Designed to run cleanly on CPU-only machines (e.g., IdeaPad 3 11th Gen i7 laptop).
"""

import os
import sys
import argparse
import json
import csv
import time
import shutil
import numpy as np

# Suppress verbose TensorFlow C++ logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure server and client directories are in sys.path for direct imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
CLIENT_DIR = os.path.join(PROJECT_ROOT, "client")

if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

import server.model as server_model_mod
sys.modules['model'] = server_model_mod

import tensorflow as tf
from client.model import Model as ClientModel
from server.model import Model as ServerModel
from server.selector import get_selector_by_name
from server.aggregator import get_aggregator_by_name, FedFV
from server.rl_env import FederatedEnv


def plot_metrics(round_history, output_dir):
    """Generates and saves research baseline performance plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")

    if not round_history:
        return

    rounds = [x["round"] for x in round_history]

    # 1. Loss vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [x.get("loss", 0.0) for x in round_history], marker='o', color='crimson', label='Global Loss')
    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.title("Global Test Loss vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_vs_round.png"))
    plt.close()

    # 2. Accuracy vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [x.get("accuracy", 0.0) for x in round_history], marker='o', color='royalblue', label='Accuracy')
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title("Global Test Accuracy vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_round.png"))
    plt.close()

    # 3. F1 & ROC-AUC vs Round
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [x.get("f1", 0.0) for x in round_history], marker='o', color='purple', label='F1 Score')
    plt.plot(rounds, [x.get("roc_auc", 0.5) for x in round_history], marker='s', color='darkorange', label='ROC AUC')
    plt.xlabel("Federated Round")
    plt.ylabel("Score")
    plt.title("F1 Score & ROC AUC vs Federated Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_vs_round.png"))
    plt.close()

    # 4. Confusion Matrix (Latest Round)
    latest = round_history[-1]
    count_keys = ["tn", "fp", "fn", "tp"]
    if all(k in latest for k in count_keys):
        cm = np.array([[int(latest["tn"]), int(latest["fp"])],
                       [int(latest["fn"]), int(latest["tp"])]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal (0)', 'AFib (1)'],
                    yticklabels=['Normal (0)', 'AFib (1)'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Global Confusion Matrix (Round {latest.get('round', '')})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_latest.png"))
        plt.close()

    print(f"[Research Runner] Metric plots saved to {output_dir}")


def run_simulation(args):
    print("=" * 60)
    print(" SmartMonitoring Decoupled Research Baseline Simulation ")
    print("=" * 60)
    print(f" Data Directory    : {args.data_dir}")
    print(f" Total Clients (N) : {args.num_clients}")
    print(f" Selected per Round: {args.select_k}")
    print(f" Total Rounds (R)  : {args.rounds}")
    print(f" Local Epochs (E)  : {args.local_epochs}")
    print(f" Batch Clients     : {args.batch_clients}")
    print(f" Aggregator        : {args.aggregator}")
    print(f" Selector          : {args.selector}")
    print(f" Output Directory  : {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_model_dir = os.path.join(args.output_dir, "tmp_models")
    os.makedirs(tmp_model_dir, exist_ok=True)

    # 1. Discover client data files
    client_files = {}
    for i in range(args.num_clients):
        cid = f"client_{i}"
        filename = f"{cid}.npz"
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Client dataset file not found: {filepath}")
        client_files[cid] = filepath

    client_ids = list(client_files.keys())
    client_id_map = {cid: i for i, cid in enumerate(client_ids)}

    # 2. Build RL Environment profiles
    profiles = {}
    for i in range(100):
        profiles[i] = {
            "cpu_frequency": 2.0e9,
            "tx_power": 0.2,
            "r_trans": 15e6
        }
    env = FederatedEnv(profiles, model_size_bits=10_000_000)

    # 3. Instantiate Selector & Aggregator
    selector = get_selector_by_name(args.selector, env=env)
    aggregator = get_aggregator_by_name(args.aggregator)

    # 4. Initialize Global Model
    global_model_path = os.path.join(tmp_model_dir, "global_model.keras")
    initial_server_model = ServerModel()
    initial_server_model.model.save(global_model_path)
    print(f"[Research Runner] Initial global model saved to {global_model_path}")

    # Initialize client metrics cache
    client_samples = {}
    client_losses = {}
    print("[Research Runner] Inspecting client data partitions...")
    for cid, path in client_files.items():
        cm = ClientModel(path, batch_size=args.batch_size)
        client_samples[cid] = cm.get_samples()
        client_losses[cid] = 1.0
        del cm
    tf.keras.backend.clear_session()

    round_history = []
    rounds_left = args.rounds

    # 5. Main Execution Loop
    for r in range(1, args.rounds + 1):
        rounds_left -= 1
        print(f"\n--- Round {r}/{args.rounds} ---")
        start_round_time = time.time()

        context = {
            "round": r,
            "rounds_left": rounds_left,
            "env": env,
            "client_id_map": client_id_map,
            "client_samples": client_samples,
            "client_losses": client_losses
        }

        # Select clients
        selected_ids = selector.select_clients(client_ids, args.select_k, context=context)
        print(f"Selected Clients for Round {r}: {selected_ids}")

        # Check if hierarchical meta-controller selected a new aggregation strategy
        if hasattr(selector, "last_chosen_agg") and selector.last_chosen_agg:
            chosen_strat = selector.last_chosen_agg
            print(f"[Hierarchical Selector] Active strategy updated to: {chosen_strat}")
            aggregator = get_aggregator_by_name(chosen_strat)

        client_data = []

        # Local training (chunked/sequential execution for CPU safety)
        for idx_start in range(0, len(selected_ids), args.batch_clients):
            chunk_cids = selected_ids[idx_start:idx_start + args.batch_clients]

            for cid in chunk_cids:
                npz_path = client_files[cid]
                client_model = ClientModel(npz_path, batch_size=args.batch_size)
                client_model.model.load_weights(global_model_path)

                if aggregator.mode == "gradients":
                    # FedFV style gradient-based training
                    t0 = time.time()
                    avg_grads, local_loss = client_model.train_local_gradients_fv()
                    comp_lat = time.time() - t0
                    measured_energy = comp_lat * 5.0
                    num_id = client_id_map[cid]
                    n_samples = client_samples[cid]

                    client_losses[cid] = float(local_loss)
                    client_data.append((avg_grads, n_samples, local_loss, num_id, comp_lat, measured_energy))
                    print(f" Client {cid}: Trained (Gradients) | Loss: {local_loss:.4f} | Latency: {comp_lat:.2f}s")

                else:
                    # Weight-based training (FedAvg, FedProx, FedAdam, Krum, SCAFFOLD, etc.)
                    t0 = time.time()
                    client_model.train(epochs=args.local_epochs, verbose=0)
                    comp_lat = time.time() - t0
                    measured_energy = comp_lat * 5.0

                    eval_res = client_model.evaluate()
                    local_loss = eval_res.get("loss", 1.0)
                    client_losses[cid] = float(local_loss)

                    local_weights_path = os.path.join(tmp_model_dir, f"{cid}_weights.keras")
                    client_model.model.save(local_weights_path)
                    n_samples = client_samples[cid]

                    client_data.append((local_weights_path, n_samples, local_loss, cid, comp_lat, measured_energy))
                    print(f" Client {cid}: Trained (Weights) | Loss: {local_loss:.4f} | Accuracy: {eval_res.get('accuracy', 0.0):.4f} | Latency: {comp_lat:.2f}s")

                del client_model

            # Clean memory after processing chunk
            tf.keras.backend.clear_session()

        # Aggregation Phase
        print(f"[Aggregator ({aggregator.__class__.__name__})] Aggregating client updates...")
        if aggregator.mode == "gradients":
            assert isinstance(aggregator, FedFV)
            global_gt = aggregator.aggregate(client_data, global_model_path, current_round=r, ModelClass=ServerModel)
            
            # Apply global gradients back to global model
            gm = ServerModel()
            gm.model.load_weights(global_model_path)
            
            # Apply update using ServerModel's trainable variables
            trainable_vars = gm.model.trainable_variables
            for var, gg in zip(trainable_vars, global_gt):
                var.assign(var.numpy() - (0.001 * gg))
            gm.model.save(global_model_path)
            del gm
        else:
            aggregator.aggregate(client_data, global_model_path, current_round=r)

        tf.keras.backend.clear_session()
        elapsed_round = time.time() - start_round_time

        # Evaluation Phase across all selected (or available) clients
        print(f"[Evaluation] Evaluating updated global model on selected clients...")
        eval_results = []
        for cid in selected_ids:
            cm = ClientModel(client_files[cid], batch_size=args.batch_size)
            cm.model.load_weights(global_model_path)
            e_res = cm.evaluate()
            eval_results.append({
                "client_id": cid,
                "samples": client_samples[cid],
                "metrics": e_res
            })
            del cm

        tf.keras.backend.clear_session()

        # Compute weighted global metrics
        total_eval_samples = sum(ev["samples"] for ev in eval_results)
        metric_names = eval_results[0]["metrics"].keys()
        round_metrics = {"round": r}
        count_metrics = {"tp", "fp", "tn", "fn"}

        for m_key in metric_names:
            if m_key in count_metrics:
                round_metrics[m_key] = int(sum(ev["metrics"][m_key] for ev in eval_results))
            else:
                w_avg = sum(ev["metrics"][m_key] * (ev["samples"] / total_eval_samples) for ev in eval_results)
                round_metrics[m_key] = float(w_avg)

        # Track system metrics
        latencies = [cd[4] for cd in client_data]
        energies = [cd[5] for cd in client_data]
        round_metrics["avg_comp_latency"] = float(np.mean(latencies)) if latencies else 0.0
        round_metrics["total_round_energy"] = float(np.sum(energies)) if energies else 0.0

        round_history.append(round_metrics)

        print(f" Round {r} Results | Loss: {round_metrics['loss']:.4f} | Acc: {round_metrics['accuracy']:.4f} | F1: {round_metrics['f1']:.4f} | Time: {elapsed_round:.2f}s")

        # Update Client Selector Policy (RL / Contextual Bandits)
        prev_loss = round_history[-2]["loss"] if len(round_history) > 1 else round_metrics["loss"]
        global_loss_delta = prev_loss - round_metrics["loss"]

        round_summary = {
            "round": r,
            "rounds_left": rounds_left,
            "selected_ids": selected_ids,
            "active_clients": client_ids,
            "client_id_map": client_id_map,
            "client_samples": client_samples,
            "client_losses": client_losses,
            "global_loss_delta": global_loss_delta,
            "local_losses": [client_losses[cid] for cid in selected_ids],
            "elapsed_round": elapsed_round,
            "client_roundtrips": {cid: elapsed_round for cid in selected_ids},
            "client_latencies": {cid: latencies[i] for i, cid in enumerate(selected_ids)},
            "client_energies": {cid: energies[i] for i, cid in enumerate(selected_ids)}
        }

        selector.update_policy(round_summary)

        # Save metrics to JSON and plot
        metrics_json_path = os.path.join(args.output_dir, "round_history.json")
        with open(metrics_json_path, "w") as f:
            json.dump(round_history, f, indent=2)

        plot_metrics(round_history, args.output_dir)

    # Cleanup temporary models directory
    if os.path.exists(tmp_model_dir):
        shutil.rmtree(tmp_model_dir)

    print("\n" + "=" * 60)
    print(f" Research Simulation Completed Successfully! ")
    print(f" Logs & Plots saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone In-Process FL Research Baseline Simulation")
    parser.add_argument("--data-dir", type=str, default="data/iid", help="Path to client dataset partitions folder")
    parser.add_argument("-n", "--num-clients", type=int, default=10, help="Total clients N (default: 10)")
    parser.add_argument("-k", "--select-k", type=int, default=3, help="Clients selected per round K (default: 3)")
    parser.add_argument("-r", "--rounds", type=int, default=5, help="Total FL rounds (default: 5)")
    parser.add_argument("-e", "--local-epochs", type=int, default=1, help="Local training epochs per round (default: 1)")
    parser.add_argument("-b", "--batch-clients", type=int, default=1, help="Clients trained per sequential batch (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size (default: 32)")
    parser.add_argument("-a", "--aggregator", type=str, default="fedavg",
                        help="Aggregation strategy: fedavg, qfedavg, fedfv, fedadam, fedprox, krum, scaffold")
    parser.add_argument("-s", "--selector", type=str, default="random",
                        help="Selection strategy: random, linucb, wls-ts, dqn, hierarchical")
    parser.add_argument("-o", "--output-dir", type=str, default="results/research_baseline",
                        help="Output directory for metrics and plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parsed_args = parser.parse_args()
    np.random.seed(parsed_args.seed)
    tf.random.set_seed(parsed_args.seed)

    run_simulation(parsed_args)
