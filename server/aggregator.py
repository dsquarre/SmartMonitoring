from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from model import Model
import copy
import math

def filter_poisoned_clients(client_data: List, global_model_path: str, mode: str = "weights", threshold_multiplier: float = 3.0) -> List:
    """
    Filters out extreme outlier/poisoned client updates dynamically using Median Absolute Deviation (MAD).
    Allows genuine variance by targeting Z-scores higher than threshold_multiplier.
    """
    if len(client_data) <= 2:
        return client_data # Need at least 3 clients for reliable median/MAD statistics
        
    norms = []
    
    if mode == "weights":
        global_model = Model()
        global_model.model.load_weights(global_model_path)
        global_weights = global_model.model.get_weights()
        
        for item in client_data:
            model_path = item[0]
            local_model = Model()
            local_model.model.load_weights(model_path)
            local_weights = local_model.model.get_weights()
            
            diffs = [lw - gw for lw, gw in zip(local_weights, global_weights)]
            flat_diff = np.concatenate([d.flatten() for d in diffs])
            norms.append(float(np.linalg.norm(flat_diff)))
            
    elif mode == "gradients":
        for item in client_data:
            client_grads = item[0]
            flat_grad = np.concatenate([g.flatten() for g in client_grads])
            norms.append(float(np.linalg.norm(flat_grad)))
            
    norms = np.array(norms)
    median_norm = np.median(norms)
    mad = np.median(np.abs(norms - median_norm))
    
    if mad < 1e-8:
        mad = 1e-8
        
    filtered_data = []
    for idx, item in enumerate(client_data):
        norm = norms[idx]
        z_score = (norm - median_norm) / mad
        client_id = item[3]
        
        if z_score > threshold_multiplier:
            print(f"[POISONING FILTER] Discarded update from client {client_id}: norm={norm:.6f}, median={median_norm:.6f}, Z-score={z_score:.2f}")
        else:
            filtered_data.append(item)
            
    if len(filtered_data) == 0:
        print("[POISONING FILTER] Warning: All client updates were filtered out. Falling back to using all updates.")
        return client_data
        
    return filtered_data

class ModelAggregator(ABC):
    @property
    @abstractmethod
    def mode(self):
        """Returns 'weights' or 'gradients' depending on strategy workflow needs."""
        pass
    @abstractmethod
    def aggregate(self, client_data: List[Tuple[str, float, float, str]], global_model_path: str, current_round: int):
        """
        Aggregate local client models and save the new global model.

        Args:
            client_data: List of tuples containing (local_model_filepath, number_of_samples, loss, client_id).
            global_model_path: Filepath where the new global model should be saved.
            current_round: The current federated learning round index.
        """
        pass

class FedAvg(ModelAggregator):
    """
    Standard Federated Averaging (FedAvg) aggregation strategy.
    """
    @property
    def mode(self):
        return "weights"
        
    def aggregate(self, client_data, global_model_path, current_round):
        print("Using FedAvg")
        client_data = filter_poisoned_clients(client_data, global_model_path, mode="weights")
        global_model = Model()
        total_samples = sum(samples for _, samples, *_, _ in client_data)

        aggregated_weights = None
        for model_path, client_samples, *_, _ in client_data:
            local_model = Model()
            local_model.model.load_weights(model_path)
            local_weights = local_model.model.get_weights()

            if aggregated_weights is None:
                aggregated_weights = [np.zeros_like(layer) for layer in local_weights]

            for i in range(len(local_weights)):
                aggregated_weights[i] += local_weights[i] * (client_samples / total_samples)

        global_model.model.set_weights(aggregated_weights)
        global_model.model.save(global_model_path)

class qFedAvg(ModelAggregator):

    def __init__(self, q=0.5):
        self.q = q
    @property
    def mode(self):
        return "weights"

    def aggregate(self, client_data, global_model_path, current_round):
        client_data = filter_poisoned_clients(client_data, global_model_path, mode="weights")
        global_model = Model()
        raw_weights = []

        for item in client_data:
            model_path, samples, loss, client_id = item[0], item[1], item[2], item[3]
            raw_weight = (samples *((loss + 1e-10) ** self.q))
            raw_weights.append(raw_weight)
        total_weight = sum(raw_weights)
        aggregated_weights = None

        print("Using qFedAvg")
        for idx, item in enumerate(client_data):
            model_path, samples, loss, client_id = item[0], item[1], item[2], item[3]
            client_weight = (raw_weights[idx] / total_weight)
            local_model = Model()
            local_model.model.load_weights(model_path)
            local_weights = (local_model.model.get_weights())
            if aggregated_weights is None:
                aggregated_weights = [np.zeros_like(layer) for layer in local_weights]
            for i in range(len(local_weights)):
                aggregated_weights[i] += (local_weights[i]*client_weight)
        global_model.model.set_weights(aggregated_weights)
        global_model.model.save(global_model_path)

class FedFV(ModelAggregator):
    def __init__(self, num_clients=10, alpha=0.1, tau=1):
        self.alpha = alpha
        self.tau = tau
        self.num_clients = num_clients
        
        # Historical memory buffers for external conflict checking across sequential rounds
        self.client_grad_history = {}
        self.client_last_round = {}

    @property
    def mode(self):
        return "gradients"

    def grad_dot(self, g1, g2):
        return sum(np.sum(a * b) for a, b in zip(g1, g2))

    def grad_norm(self, g):
        return np.sqrt(sum(np.sum(layer * layer) for layer in g))

    def grad_scale(self, g, scalar):
        return [layer * scalar for layer in g]
        
    def grad_sub(self, g1, g2):
        return [a - b for a, b in zip(g1, g2)]

    def aggregate(self, client_data, global_model_path, current_round, ModelClass):
        """
        Executes internal/external conflict resolution routines directly on raw gradient lists.
        """
        client_data = filter_poisoned_clients(client_data, global_model_path, mode="gradients")
        gradients = []
        losses = []
        client_ids = []
        
        for (client_grads, samples, loss, numeric_id, *_, _) in client_data:
            gradients.append(client_grads)
            losses.append(loss)
            client_ids.append(numeric_id)
            
            # FIX 2: Safe dictionary assignment (will never throw an IndexError!)
            self.client_grad_history[numeric_id] = client_grads
            self.client_last_round[numeric_id] = current_round

        # Sort clients based on loss trajectory
        loop_indices = list(range(len(losses)))
        sorted_pairs = sorted(zip(losses, loop_indices), key=lambda x: x[0])
        sorted_order = [x[1] for x in sorted_pairs]

        # Alpha Protection Tail Boundary calculation
        keep_original_loop_indices = []
        if self.alpha > 0 and len(sorted_order) > 1:
            boundary = math.ceil((len(sorted_order) - 1) * (1 - self.alpha))
            keep_original_loop_indices = sorted_order[boundary:]

        # Internal Conflict Mitigation Loop
        projected_grads = copy.deepcopy(gradients)
        for i in range(len(projected_grads)):
            if i in keep_original_loop_indices:
                continue
            for j in sorted_order:
                if i == j:
                    continue
                dot = self.grad_dot(projected_grads[i], gradients[j])
                if dot < 0:
                    denom = (self.grad_norm(gradients[j]) ** 2) + 1e-12
                    correction = self.grad_scale(gradients[j], dot / denom)
                    projected_grads[i] = self.grad_sub(projected_grads[i], correction)

        # Average projected gradients
        global_model = ModelClass()
        global_model.model.load_weights(global_model_path)
        
        # FIX 1: Base your shape blueprints on trainable_variables, NOT get_weights()
        global_trainable_vars = global_model.model.trainable_variables

        # Step 1: Create a clean base tracking structure using the correct trainable shapes
        gt = [np.zeros(var.shape, dtype=np.float32) for var in global_trainable_vars]
        
        # Step 2: Sum the projected gradients layer-by-layer across all clients
        num_projected = len(projected_grads)
        for pg in projected_grads:
            for layer_idx in range(len(global_trainable_vars)):
                # Perform pure, element-wise addition matching identical shape slots
                gt[layer_idx] = gt[layer_idx] + (pg[layer_idx] / num_projected)
        
        # External Conflict Mitigation Loop (Tau History Lookup Window)
        if current_round >= self.tau:
            for k in range(self.tau - 1, -1, -1):
                gcs = []
                
                for cid in self.client_last_round.keys():
                    if self.client_last_round[cid] == (current_round - k):
                        hist_grad = self.client_grad_history.get(cid)
                        if hist_grad is not None:
                            if self.grad_dot(gt, hist_grad) < 0:
                                gcs.append(hist_grad)
                
                if gcs:
                    # FIX 2: Align the historical constraint matrices to trainable variables as well
                    g_con = [np.zeros(var.shape, dtype=np.float32) for var in global_trainable_vars]
                    for hist_g in gcs:
                        for layer_idx in range(len(global_trainable_vars)):
                            g_con[layer_idx] = g_con[layer_idx] + hist_g[layer_idx]
                            
                    dot_ext = self.grad_dot(gt, g_con)
                    if dot_ext < 0:
                        denom_ext = (self.grad_norm(g_con) ** 2) + 1e-12
                        correction_ext = self.grad_scale(g_con, dot_ext / denom_ext)
                        gt = self.grad_sub(gt, correction_ext)

        # Rescaling projection norm matching
        # FIX 3: Align the baseline norm normalization tracking list
        original_avg = [np.zeros(var.shape, dtype=np.float32) for var in global_trainable_vars]
        num_gradients = len(gradients)
        for g in gradients:
            for layer_idx in range(len(global_trainable_vars)):
                original_avg[layer_idx] = original_avg[layer_idx] + (g[layer_idx] / num_gradients)
                
        gnorm = self.grad_norm(original_avg)
        gt_norm = self.grad_norm(gt)
        if gt_norm > 0:
            gt = self.grad_scale(gt, gnorm / gt_norm)

        return gt

class FedAdam(ModelAggregator):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.previous_global_model = ("models/global_model_0.keras")
        self.m = None
        self.v = None
        self.t = 0

    @property
    def mode(self):
        return "weights"

    def aggregate(self,client_data,global_model_path,current_round):
        if self.previous_global_model is None:
            self.previous_global_model = global_model_path
            global_model = Model()
            global_model.model.save( global_model_path)
            return

        client_data = filter_poisoned_clients(client_data, self.previous_global_model, mode="weights")
        global_model = Model()
        print("Using FedAdam")

        global_model.model.load_weights(self.previous_global_model)
        global_weights = (global_model.model.get_weights())
        total_samples = sum(samples for _, samples, *_, _ in client_data)
        aggregated_gradient = [np.zeros_like(layer) for layer in global_weights]

        for (model_path,samples,loss,client_id, *_, _) in client_data:
            local_model = Model()
            local_model.model.load_weights(model_path)
            local_weights = (local_model.model.get_weights())
            client_weight = (samples / total_samples)

            for i in range(len(global_weights)):
                grad = (local_weights[i]-global_weights[i])
                aggregated_gradient[i] += ( client_weight * grad)

        if self.m is None:

            self.m = [ np.zeros_like(layer) for layer in aggregated_gradient]
            self.v = [np.zeros_like(layer) for layer in aggregated_gradient]

        self.t += 1
        new_weights = []
        for i in range(len(global_weights)):
            g = aggregated_gradient[i]
            self.m[i] = (self.beta1* self.m[i]+(1 - self.beta1)* g)
            self.v[i] = (self.beta2* self.v[i]+(1 - self.beta2)* np.square(g))
            m_hat = (self.m[i]/(1-self.beta1 ** self.t))
            v_hat = (self.v[i]/(1-self.beta2 ** self.t))
            update = (self.lr*m_hat/(np.sqrt(v_hat)+self.epsilon))
            new_weights.append(global_weights[i]+update)

        global_model.model.set_weights(new_weights)
        global_model.model.save(global_model_path)
        self.previous_global_model = (global_model_path)

class FedProx(ModelAggregator):
    """
    FedProx aggregation strategy with proximal regularization penalty.
    """
    def __init__(self, mu: float = 0.01):
        self.mu = mu

    @property
    def mode(self):
        return "weights"

    def aggregate(self, client_data, global_model_path, current_round):
        print("Using FedProx")
        client_data = filter_poisoned_clients(client_data, global_model_path, mode="weights")
        global_model = Model()
        global_model.model.load_weights(global_model_path)
        global_weights = global_model.model.get_weights()
        total_samples = sum(samples for _, samples, *_, _ in client_data)

        if total_samples == 0 or len(client_data) == 0:
            return

        aggregated_weights = [np.zeros_like(layer) for layer in global_weights]
        for model_path, client_samples, *_, _ in client_data:
            local_model = Model()
            local_model.model.load_weights(model_path)
            local_weights = local_model.model.get_weights()
            
            # Weight sample ratio adjusted by proximal drift penalty
            drift = sum(np.linalg.norm(lw - gw) for lw, gw in zip(local_weights, global_weights))
            drift_penalty = 1.0 / (1.0 + self.mu * drift)
            effective_weight = (client_samples / total_samples) * drift_penalty
            
            for i in range(len(local_weights)):
                aggregated_weights[i] += local_weights[i] * effective_weight
                
        global_model.model.set_weights(aggregated_weights)
        global_model.model.save(global_model_path)

class Krum(ModelAggregator):
    """
    Krum / Multi-Krum Byzantine-resilient aggregation strategy.
    Selects update that minimizes Euclidean distance to its closest neighbors.
    """
    def __init__(self, num_byzantine: int = 1):
        self.num_byzantine = num_byzantine

    @property
    def mode(self):
        return "weights"

    def aggregate(self, client_data, global_model_path, current_round):
        print("Using Krum")
        client_data = filter_poisoned_clients(client_data, global_model_path, mode="weights")
        n = len(client_data)
        if n == 0:
            return
        
        flat_client_weights = []
        all_local_weights = []
        for model_path, *_, _ in client_data:
            local_model = Model()
            local_model.model.load_weights(model_path)
            l_weights = local_model.model.get_weights()
            all_local_weights.append(l_weights)
            flat = np.concatenate([w.flatten() for w in l_weights])
            flat_client_weights.append(flat)

        f = min(self.num_byzantine, max(0, (n - 3) // 2))
        k = max(1, n - f - 2)

        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(flat_client_weights[i] - flat_client_weights[j])
                    distances.append(dist)
            distances.sort()
            score = sum(distances[:k])
            scores.append(score)

        best_idx = int(np.argmin(scores))
        selected_weights = all_local_weights[best_idx]

        global_model = Model()
        global_model.model.set_weights(selected_weights)
        global_model.model.save(global_model_path)

class SCAFFOLD(ModelAggregator):
    """
    SCAFFOLD aggregation strategy using control variate variance reduction.
    """
    def __init__(self, lr: float = 1.0):
        self.lr = lr
        self.c_global = None  # Global control variate
        self.c_clients = {}   # Dict[client_id, control variate]

    @property
    def mode(self):
        return "weights"

    def aggregate(self, client_data, global_model_path, current_round):
        print("Using SCAFFOLD")
        client_data = filter_poisoned_clients(client_data, global_model_path, mode="weights")
        global_model = Model()
        global_model.model.load_weights(global_model_path)
        global_weights = global_model.model.get_weights()

        if self.c_global is None:
            self.c_global = [np.zeros_like(layer) for layer in global_weights]

        n = len(client_data)
        if n == 0:
            return

        total_samples = sum(samples for _, samples, *_, _ in client_data)
        aggregated_weights = [np.zeros_like(layer) for layer in global_weights]
        delta_c_sum = [np.zeros_like(layer) for layer in global_weights]

        for item in client_data:
            model_path, samples, loss, cid = item[0], item[1], item[2], item[3]
            local_model = Model()
            local_model.model.load_weights(model_path)
            local_weights = local_model.model.get_weights()

            if cid not in self.c_clients:
                self.c_clients[cid] = [np.zeros_like(layer) for layer in global_weights]

            c_i = self.c_clients[cid]
            c_i_new = []
            for layer_idx in range(len(global_weights)):
                delta_w = global_weights[layer_idx] - local_weights[layer_idx]
                c_i_layer = c_i[layer_idx] - self.c_global[layer_idx] + (1.0 / self.lr) * delta_w
                c_i_new.append(c_i_layer)
                delta_c_sum[layer_idx] += (c_i_layer - c_i[layer_idx]) / n
            
            self.c_clients[cid] = c_i_new
            client_weight = samples / total_samples
            for i in range(len(local_weights)):
                aggregated_weights[i] += local_weights[i] * client_weight

        for i in range(len(global_weights)):
            self.c_global[i] += delta_c_sum[i]

        global_model.model.set_weights(aggregated_weights)
        global_model.model.save(global_model_path)

def get_aggregator_by_name(name: str, **kwargs) -> ModelAggregator:
    """
    Factory function returning an instance of the requested ModelAggregator strategy.
    Supported names: 'FedAvg', 'qFedAvg', 'FedFV', 'FedAdam', 'FedProx', 'Krum', 'SCAFFOLD'
    """
    name_lower = name.lower()
    if name_lower == "fedavg":
        return FedAvg()
    elif name_lower == "qfedavg":
        return qFedAvg(q=kwargs.get("q", 0.5))
    elif name_lower == "fedfv":
        return FedFV(num_clients=kwargs.get("num_clients", 10))
    elif name_lower == "fedadam":
        return FedAdam(lr=kwargs.get("lr", 0.001))
    elif name_lower == "fedprox":
        return FedProx(mu=kwargs.get("mu", 0.01))
    elif name_lower == "krum":
        return Krum(num_byzantine=kwargs.get("num_byzantine", 1))
    elif name_lower == "scaffold":
        return SCAFFOLD(lr=kwargs.get("lr", 1.0))
    else:
        raise ValueError(f"Unknown aggregator strategy name: {name}")