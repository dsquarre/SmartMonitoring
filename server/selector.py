from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random
import numpy as np

class ClientSelector(ABC):
    @abstractmethod
    def select_clients(self, client_ids: List[str], k: int, context: Dict[str, Any] = None) -> List[str]:
        """
        Select k clients out of the available clients.

        Args:
            client_ids: List of connected client IDs.
            k: Number of clients to select.
            context: Dictionary containing extra context (e.g., round number, metrics history).

        Returns:
            List of selected client IDs.
        """
        pass

class RandomClientSelector(ClientSelector):
    """
    Selects k clients uniformly at random from the connected clients.
    """
    def select_clients(self, client_ids: List[str], k: int, context: Dict[str, Any] = None) -> List[str]:
        if not client_ids:
            return []
        k = min(k, len(client_ids))
        return random.sample(client_ids, k)

class BaseRLAgent(ABC):
    @abstractmethod
    def get_action(self, state: np.ndarray, num_clients: int, k: int) -> List[int]:
        """
        Returns a list of k selected client indices based on state representation.
        """
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray):
        """
        Train the RL agent.
        """
        pass

class RandomRLAgent(BaseRLAgent):
    """A baseline Random Agent that fits the interface."""
    def get_action(self, state: np.ndarray, num_clients: int, k: int) -> List[int]:
        indices = list(range(num_clients))
        return list(np.random.choice(indices, size=k, replace=False))

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray):
        pass

class RLClientSelector(ClientSelector):
    def __init__(self, agent: BaseRLAgent, env: Any):
        self.agent = agent
        self.env = env
        self.last_state = None
        self.last_action = None

    def select_clients(self, client_ids: List[str], k: int, context: Dict[str, Any] = None) -> List[str]:
        if not client_ids:
            return []
        
        context = context or {}
        # Construct state vector
        state = self._build_state(client_ids, context)
        self.last_state = state

        # Get action from agent (returns indices)
        selected_indices = self.agent.get_action(state, len(client_ids), k)
        self.last_action = selected_indices

        selected_ids = [client_ids[idx] for idx in selected_indices]
        return selected_ids

    def _build_state(self, client_ids: List[str], context: Dict[str, Any]) -> np.ndarray:
        state_list = []
        for i, cid in enumerate(client_ids):
            # Numeric ID mapping
            num_id = context.get("client_id_map", {}).get(cid, i)
            profile = self.env.profiles.get(num_id, {"cpu_frequency": 2.0e9})
            
            # Dynamic features
            samples = context.get("client_samples", {}).get(cid, 0)
            last_loss = context.get("client_losses", {}).get(cid, 1.0)
            
            state_list.append([
                float(samples), 
                float(last_loss), 
                float(profile["cpu_frequency"])
            ])
        return np.array(state_list, dtype=np.float32)
