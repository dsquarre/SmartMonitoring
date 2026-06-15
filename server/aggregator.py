from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from model import Model
import copy
import math

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
        global_model = Model()
        total_samples = sum(samples for _, samples, _, _ in client_data)

        aggregated_weights = None
        for model_path, client_samples, _, _ in client_data:
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
        global_model = Model()
        raw_weights = []

        for (model_path, samples, loss, client_id) in client_data:
            raw_weight = (samples *((loss + 1e-10) ** self.q))
            raw_weights.append(raw_weight)
        total_weight = sum(raw_weights)
        aggregated_weights = None

        print("Using qFedAvg")
        for idx, (model_path, samples, loss, client_id) in enumerate(client_data):
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
        gradients = []
        losses = []
        client_ids = []
        
        for (client_grads, samples, loss, numeric_id) in client_data:
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

    def aggregate(self,client_data,global_model_path,current_round):
        if self.previous_global_model is None:
            self.previous_global_model = global_model_path
            global_model = Model()
            global_model.model.save( global_model_path)
            return

        global_model = Model()
        print("Using FedAdam")

        global_model.model.load_weights(self.previous_global_model)
        global_weights = (global_model.model.get_weights())
        total_samples = sum(samples for _, samples, _, _ in client_data)
        aggregated_gradient = [np.zeros_like(layer) for layer in global_weights]

        for (model_path,samples,loss,client_id) in client_data:
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