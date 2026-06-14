from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from model import Model
import copy

class ModelAggregator(ABC):
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

    def __init__(self):
        self.client_grad_history = {}
        self.client_last_round = {}
        self.previous_global_model = ("models/global_model_0.keras")

    def grad_dot(self, g1, g2):
        return sum(np.sum(a * b) for a, b in zip(g1, g2))

    def grad_norm(self, g):
        return np.sqrt(sum(np.sum(layer * layer) for layer in g))

    def grad_scale(self, g, scalar):
        return [layer * scalar for layer in g]
        
    def grad_sub(self, g1, g2):
        return [a - b for a, b in zip(g1, g2)]

    def grad_add(self, g1, g2):
        return [a + b for a, b in zip(g1, g2)]
        
    def aggregate(self, client_data, global_model_path, current_round):
        if len(client_data) == 0:
            return
        global_model = Model()
        print("Using FedFV")

        global_model.model.load_weights(self.previous_global_model)
        global_weights = ( global_model.model.get_weights())
        gradients = []
        losses = []
        sample_weights = []
        total_samples = sum(samples for _, samples, _, _ in client_data)

        for (model_path, samples, loss, client_id) in client_data:
            local_model = Model()
            local_model.model.load_weights(model_path)
            local_weights = (local_model.model.get_weights())
            grad = []
            for gw, lw in zip(global_weights,local_weights):
                # gradient = global - local
                grad.append(gw - lw)
            gradients.append(grad)
            losses.append(loss)
            sample_weights.append(samples / total_samples)
            self.client_grad_history[client_id] = grad
            self.client_last_round[client_id] = current_round

        order = sorted(range(len(losses)),key=lambda i: losses[i])

        # Internal conflict mitigation
        projected_grads = copy.deepcopy(gradients)
        for i in range(len(projected_grads)):
            for j in order:
                if i == j:
                    continue
                dot = self.grad_dot(projected_grads[i],gradients[j])
                if dot < 0:
                    denom = (self.grad_norm(gradients[j]) ** 2) + 1e-12
                    coeff = dot / denom
                    correction = (self.grad_scale(gradients[j],coeff))
                    projected_grads[i] = (self.grad_sub(projected_grads[i],correction))
        #aggregated gradients
        aggregated_gradient = []
        for layer_idx in range(len(global_weights)):
            agg_layer = np.zeros_like(global_weights[layer_idx])
            for client_idx in range(len(projected_grads)):
                agg_layer += (projected_grads[client_idx][layer_idx]*sample_weights[client_idx])
            aggregated_gradient.append(agg_layer)
        #original average
        original_avg = []
        for layer_idx in range(len(global_weights)):
            layer = np.zeros_like(global_weights[layer_idx])
            for client_idx in range(len(gradients)):
                layer += (gradients[client_idx][layer_idx]*sample_weights[client_idx])
            original_avg.append(layer)

        original_norm = (self.grad_norm(original_avg))
        projected_norm = (self.grad_norm(aggregated_gradient))
        if projected_norm > 0:
            scale = (original_norm/projected_norm)
            aggregated_gradient = [layer * scale for layer in (aggregated_gradient)]
        #new average
        new_weights = []
        for gw, grad in zip(global_weights, aggregated_gradient):
            # global = global - gradient
            new_weights.append(gw - grad )

        global_model.model.set_weights(new_weights)
        global_model.model.save(global_model_path)
        self.previous_global_model = (global_model_path)

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