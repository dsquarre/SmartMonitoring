from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from model import Model

class ModelAggregator(ABC):
    @abstractmethod
    def aggregate(self, client_data: List[Tuple[str, float]], global_model_path: str):
        """
        Aggregate local client models and save the new global model.

        Args:
            client_data: List of tuples containing (local_model_filepath, number_of_samples).
            global_model_path: Filepath where the new global model should be saved.
        """
        pass

class FedAvg(ModelAggregator):
    """
    Standard Federated Averaging (FedAvg) aggregation strategy.
    """
    def aggregate(self, client_data: List[Tuple[str, float]], global_model_path: str):
        global_model = Model()
        total_samples = sum(samples for _, samples in client_data)

        aggregated_weights = None
        for model_path, client_samples in client_data:
            local_model = Model()
            local_model.model.load_weights(model_path)
            local_weights = local_model.model.get_weights()

            if aggregated_weights is None:
                aggregated_weights = [np.zeros_like(layer) for layer in local_weights]

            for i in range(len(local_weights)):
                aggregated_weights[i] += local_weights[i] * (client_samples / total_samples)

        global_model.model.set_weights(aggregated_weights)
        global_model.model.save(global_model_path)
