from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random
import numpy as np
import itertools

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
    def get_action(self, state: np.ndarray, num_clients: int, k: int, context: Dict[str, Any] = None) -> List[int]:
        """
        Returns a list of k selected client indices based on state representation.
        """
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, context: Dict[str, Any] = None):
        """
        Train the RL agent.
        """
        pass

class RandomRLAgent(BaseRLAgent):
    """A baseline Random Agent that fits the interface."""
    def get_action(self, state: np.ndarray, num_clients: int, k: int, context: Dict[str, Any] = None) -> List[int]:
        indices = list(range(num_clients))
        return list(np.random.choice(indices, size=k, replace=False))

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, context: Dict[str, Any] = None):
        pass

class QLearningAgent(BaseRLAgent):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # Map state_str -> numpy array of size num_actions
        self.combinations_cache = {}
        self.client_cost_profiles = {}  # Cache static profiles as 'E' or 'X'

    def _get_combinations(self, num_clients, k):
        key = (num_clients, k)
        if key not in self.combinations_cache:
            self.combinations_cache[key] = list(itertools.combinations(range(num_clients), k))
        return self.combinations_cache[key]

    def _discretize_state(self, state: np.ndarray, num_clients: int, context: Dict[str, Any]) -> str:
        if state.size == 0:
            return "empty"
        
        # 1. Determine Global Stage
        current_r = context.get("round", 1)
        total_r = current_r + context.get("rounds_left", 10)
        progress = current_r / max(1, total_r)
        if progress < 0.3:
            stage = "Early"
        elif progress < 0.7:
            stage = "Mid"
        else:
            stage = "Late"

        # 2. Build Client Cost Profiles (Static classification for Energy & Latency)
        if not self.client_cost_profiles:
            costs = []
            env = context.get("env")
            if env:
                for i in range(num_clients):
                    cost = env.compute_client_cost(i, samples=1000)
                    score = cost["t_total"] * 1.0 + cost["E_total"] * 1000.0
                    costs.append((i, score))
                avg_score = np.mean([c[1] for c in costs])
                for idx, score in costs:
                    self.client_cost_profiles[idx] = "E" if score < avg_score else "X"
            else:
                for i in range(num_clients):
                    self.client_cost_profiles[i] = "E"

        # 3. Discretize Losses
        losses = state[:, 1]
        avg_loss = np.mean(losses) if len(losses) > 0 else 1.0

        client_states = []
        for i in range(num_clients):
            loss_val = state[i, 1] if i < len(losses) else 1.0
            loss_bucket = "H" if loss_val >= avg_loss else "L"
            cost_bucket = self.client_cost_profiles.get(i, "E")
            client_states.append(f"{i}:{loss_bucket}{cost_bucket}")

        return f"{stage} | " + ",".join(client_states)

    def get_action(self, state: np.ndarray, num_clients: int, k: int, context: Dict[str, Any] = None) -> List[int]:
        context = context or {}
        state_str = self._discretize_state(state, num_clients, context)
        combinations = self._get_combinations(num_clients, k)
        num_actions = len(combinations)

        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(num_actions, dtype=np.float32)

        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(num_actions)
        else:
            action_idx = int(np.argmax(self.q_table[state_str]))

        self.last_action_idx = action_idx
        self.last_state_str = state_str

        return list(combinations[action_idx])

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, context: Dict[str, Any] = None):
        state_str = getattr(self, 'last_state_str', None)
        action_idx = getattr(self, 'last_action_idx', None)
        if state_str is None or action_idx is None:
            return

        context = context or {}
        num_clients = len(state)
        next_state_str = self._discretize_state(next_state, num_clients, context)
        num_actions = len(self.q_table[state_str])
        
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(num_actions, dtype=np.float32)

        best_next_q = np.max(self.q_table[next_state_str])
        current_q = self.q_table[state_str][action_idx]
        
        self.q_table[state_str][action_idx] = current_q + self.lr * (reward + self.gamma * best_next_q - current_q)
        print(f"[Q-Learning] Updated Q table. State: {state_str[:40]}... -> Q-value: {self.q_table[state_str][action_idx]:.4f}")

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
        selected_indices = self.agent.get_action(state, len(client_ids), k, context=context)
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
