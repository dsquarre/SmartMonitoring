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
    def update_policy(self, round_summary: Dict[str, Any]):
        """
        Optional hook to update selector policy (e.g., for RL/learning-based selectors).

        Args:
            round_summary: Dictionary containing round metrics, losses, latencies, and costs.
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
        print("using random RL")
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
        print("using qlearning")
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

class LinUCBAgent(BaseRLAgent):
    """
    LinUCB Contextual Bandit Agent for fast, sample-efficient client selection.
    Learns linear reward model with Upper Confidence Bound exploration.
    """
    def __init__(self, alpha: float = 1.0, feature_dim: int = 8):
        self.alpha = alpha
        self.feature_dim = feature_dim
        # Shared Ridge Regression covariance matrix A and target vector b
        self.A = np.eye(feature_dim, dtype=np.float64)
        self.b = np.zeros((feature_dim, 1), dtype=np.float64)
        self.A_inv = np.eye(feature_dim, dtype=np.float64)
        self.recompute_inv = True

    def get_action(self, state: np.ndarray, num_clients: int, k: int, context: Dict[str, Any] = None) -> List[int]:
        print("using LinUCB contextual bandit")
        context = context or {}
        active_indices = context.get("active_indices", list(range(num_clients)))
        k = min(k, len(active_indices))
        if k == 0 or state.shape[0] == 0:
            return []

        if self.recompute_inv:
            self.A_inv = np.linalg.inv(self.A)
            self.recompute_inv = False

        theta = self.A_inv @ self.b  # (feature_dim, 1)

        scores = np.full(num_clients, -np.inf, dtype=np.float64)
        for idx in active_indices:
            x_i = state[idx].reshape(-1, 1)  # (feature_dim, 1)
            # LinUCB score = theta^T * x + alpha * sqrt(x^T * A_inv * x)
            expected_reward = float(theta.T @ x_i)
            uncertainty = float(np.sqrt(x_i.T @ self.A_inv @ x_i))
            scores[idx] = expected_reward + self.alpha * uncertainty

        # Pick top k indices with highest scores among active clients (Action Masking)
        selected_indices = np.argsort(scores)[::-1][:k].tolist()
        print(f"[LinUCB Agent] Selected top {len(selected_indices)} clients out of {len(active_indices)} active clients.")
        return selected_indices

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, context: Dict[str, Any] = None):
        context = context or {}
        vector_rewards = context.get("vector_rewards", {})  # Dict[idx, reward]
        
        for idx in range(len(state)):
            x_i = state[idx].reshape(-1, 1)
            r_i = vector_rewards.get(idx, reward if idx in action else 0.0)
            
            # Update LinUCB model parameters for clients with reward signal
            if idx in action or idx in vector_rewards:
                self.A += x_i @ x_i.T
                self.b += r_i * x_i
                self.recompute_inv = True
                
        print(f"[LinUCB Agent] Updated LinUCB model parameters A and b.")

class RLClientSelector(ClientSelector):
    def __init__(self, agent: BaseRLAgent, env: Any):
        self.agent = agent
        self.env = env
        self.last_state = None
        self.last_action = None
        self.last_client_ids = []
        
        # State tracking buffers across rounds
        self.client_staleness: Dict[str, int] = {}
        self.client_ema_latency: Dict[str, float] = {}
        self.client_ema_energy: Dict[str, float] = {}
        self.client_has_telemetry: Dict[str, float] = {}  # 1.0 if empirical metrics exist, 0.0 if default

    def select_clients(self, client_ids: List[str], k: int, context: Dict[str, Any] = None) -> List[str]:
        if not client_ids:
            return []
        
        context = context or {}
        active_clients = context.get("active_clients", client_ids)
        active_indices = [i for i, cid in enumerate(client_ids) if cid in active_clients]
        
        context["active_indices"] = active_indices
        context["env"] = self.env
        
        # Construct state vector (N x M feature matrix)
        state = self._build_state(client_ids, context)
        self.last_state = state
        self.last_client_ids = client_ids

        # Get action from agent (returns indices into client_ids)
        selected_indices = self.agent.get_action(state, len(client_ids), k, context=context)
        self.last_action = selected_indices

        # Update staleness counters
        selected_ids = [client_ids[idx] for idx in selected_indices]
        for cid in client_ids:
            if cid in selected_ids:
                self.client_staleness[cid] = 0
            else:
                self.client_staleness[cid] = self.client_staleness.get(cid, 0) + 1

        return selected_ids

    def _build_state(self, client_ids: List[str], context: Dict[str, Any]) -> np.ndarray:
        client_losses = context.get("client_losses", {})
        avg_loss = np.mean(list(client_losses.values())) if client_losses else 1.0
        
        state_list = []
        for i, cid in enumerate(client_ids):
            num_id = context.get("client_id_map", {}).get(cid, i)
            profile = self.env.profiles.get(num_id, {"cpu_frequency": 2.0e9})
            
            # Dynamic features
            samples = context.get("client_samples", {}).get(cid, 1000) / 1000.0
            last_loss = float(client_losses.get(cid, 1.0))
            loss_staleness = float(avg_loss - last_loss)
            
            # Telemetry metrics (default to 0.0 if not available per AGENTS.md rule)
            lat = float(self.client_ema_latency.get(cid, 0.0))
            eng = float(self.client_ema_energy.get(cid, 0.0))
            has_telemetry = float(self.client_has_telemetry.get(cid, 0.0))
            
            staleness = float(self.client_staleness.get(cid, 0))
            bias = 1.0
            
            state_list.append([
                last_loss,
                loss_staleness,
                lat,
                eng,
                has_telemetry,
                staleness,
                samples,
                bias
            ])
        return np.array(state_list, dtype=np.float32)

    def update_policy(self, round_summary: Dict[str, Any]):
        if self.last_state is None or self.last_action is None:
            return

        selected_ids = round_summary.get("selected_ids", [])
        client_id_map = round_summary.get("client_id_map", {})
        client_samples = round_summary.get("client_samples", {})
        client_losses = round_summary.get("client_losses", {})
        global_loss_delta = round_summary.get("global_loss_delta", 0.0)
        local_losses = round_summary.get("local_losses", [])
        active_clients = round_summary.get("active_clients", self.last_client_ids)
        roundtrips = round_summary.get("client_roundtrips", {})
        latencies = round_summary.get("client_latencies", {})
        energies = round_summary.get("client_energies", {})
        
        selected_metrics = {}
        for cid in selected_ids:
            num_id = client_id_map.get(cid, 0)
            samples = client_samples.get(cid, 1000)
            indiv_rt = roundtrips.get(cid, round_summary.get("elapsed_round"))
            comp_lat = latencies.get(cid, 1.0)
            energy = energies.get(cid, 5.0)
            cost_dict = self.env.compute_client_cost(
                num_id, samples, comp_lat, energy, indiv_rt
            )
            selected_metrics[cid] = cost_dict
            
            # Update EMA metrics and telemetry flag for selected clients
            alpha = 0.3
            self.client_ema_latency[cid] = (1 - alpha) * self.client_ema_latency.get(cid, 0.0) + alpha * cost_dict["t_total"]
            self.client_ema_energy[cid] = (1 - alpha) * self.client_ema_energy.get(cid, 0.0) + alpha * cost_dict["E_total"]
            self.client_has_telemetry[cid] = 1.0

        # Compute per-client vector rewards if supported by env
        if hasattr(self.env, "calculate_vector_rewards"):
            c_rewards, reward = self.env.calculate_vector_rewards(
                self.last_client_ids, selected_ids, selected_metrics, 
                global_loss_delta, client_losses, self.client_staleness
            )
            # Map client_id -> index reward dictionary
            vector_rewards = {i: c_rewards[cid] for i, cid in enumerate(self.last_client_ids) if cid in c_rewards}
        else:
            reward = self.env.calculate_reward(
                {client_id_map.get(cid, 0): m for cid, m in selected_metrics.items()},
                global_loss_delta, local_losses
            )
            vector_rewards = {}

        print(f"[RL Environment] Round {round_summary.get('round', 1)} Stats:")
        print(f"  - Delta Global Loss: {global_loss_delta:.4f}")
        print(f"  - Calculated Reward: {reward:.4f}")
        
        next_context = {
            "round": round_summary.get("round", 1),
            "rounds_left": round_summary.get("rounds_left", 0),
            "client_id_map": client_id_map,
            "client_samples": client_samples,
            "client_losses": client_losses,
            "active_clients": active_clients
        }
        next_state = self._build_state(self.last_client_ids, next_context)
        
        update_context = next_context.copy()
        update_context["vector_rewards"] = vector_rewards
        
        self.agent.update(self.last_state, self.last_action, reward, next_state, context=update_context)