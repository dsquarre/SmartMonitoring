from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import random
import numpy as np
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

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



class LinUCBAgent(BaseRLAgent):
    """
    Discounted-LinUCB (D-LinUCB) Contextual Bandit Agent for piecewise-stationary environments.
    Applies exponential discounting factor gamma in (0.90, 0.99) to past observations
    to continuously adapt to model loss convergence while maintaining fast O(1) execution time.
    Supports dynamic feature dimensions (e.g. 8 for standalone RL, 15 for Hierarchical FL).
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 0.95, feature_dim: int = 8):
        self.alpha = alpha
        self.gamma = gamma
        self.feature_dim = feature_dim
        self._init_dim(feature_dim)

    def _init_dim(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.A = np.eye(feature_dim, dtype=np.float64)
        self.b = np.zeros((feature_dim, 1), dtype=np.float64)
        self.A_inv = np.eye(feature_dim, dtype=np.float64)
        self.recompute_inv = True

    def get_action(self, state: np.ndarray, num_clients: int, k: int, context: Dict[str, Any] = None) -> List[int]:
        print("using Discounted-LinUCB (D-LinUCB) contextual bandit")
        context = context or {}
        active_indices = context.get("active_indices", list(range(num_clients)))
        k = min(k, len(active_indices))
        if k == 0 or state.shape[0] == 0:
            return []

        if state.ndim == 2 and state.shape[1] != self.feature_dim:
            self._init_dim(state.shape[1])

        if self.recompute_inv:
            self.A_inv = np.linalg.inv(self.A)
            self.recompute_inv = False

        theta = self.A_inv @ self.b  # (feature_dim, 1)

        scores = np.full(num_clients, -np.inf, dtype=np.float64)
        for idx in active_indices:
            x_i = state[idx].reshape(-1, 1)  # (feature_dim, 1)
            # LinUCB score = theta^T * x + alpha * sqrt(x^T * A_inv * x)
            expected_reward = float((theta.T @ x_i).item())
            uncertainty = float(np.sqrt((x_i.T @ self.A_inv @ x_i).item()))
            scores[idx] = expected_reward + self.alpha * uncertainty

        # Pick top k indices with highest scores among active clients (Action Masking)
        selected_indices = np.argsort(scores)[::-1][:k].tolist()
        print(f"[D-LinUCB Agent] Selected top {len(selected_indices)} clients out of {len(active_indices)} active clients.")
        return selected_indices

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, context: Dict[str, Any] = None):
        context = context or {}
        vector_rewards = context.get("vector_rewards", {})  # Dict[idx, reward]
        
        if state.ndim == 2 and state.shape[1] != self.feature_dim:
            self._init_dim(state.shape[1])

        # Apply exponential discounting gamma to past covariance A and target vector b for non-stationarity
        self.A = self.gamma * self.A + (1.0 - self.gamma) * np.eye(self.feature_dim, dtype=np.float64)
        self.b = self.gamma * self.b

        for idx in range(len(state)):
            x_i = state[idx].reshape(-1, 1)
            r_i = vector_rewards.get(idx, reward if idx in action else 0.0)
            
            # Update D-LinUCB model parameters for clients with reward signal
            if idx in action or idx in vector_rewards:
                self.A += x_i @ x_i.T
                self.b += r_i * x_i
                self.recompute_inv = True
                
        print(f"[D-LinUCB Agent] Updated D-LinUCB model parameters (gamma={self.gamma}).")

class WLSTSAgent(BaseRLAgent):
    """
    Weighted Least Squares Thompson Sampling (WLS-TS) Agent (Burtini et al., 2015).
    Combines Weighted Least Squares regression with Bayesian Posterior Sampling.
    Applies exponential discounting factor gamma to past observations and samples parameter vector
    tilde_theta ~ N(hat_theta, sigma^2 * A^-1) for posterior-driven exploration.
    """
    def __init__(self, gamma: float = 0.95, sigma: float = 0.25, feature_dim: int = 8):
        self.gamma = gamma
        self.sigma = sigma
        self.feature_dim = feature_dim
        self._init_dim(feature_dim)

    def _init_dim(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.A = np.eye(feature_dim, dtype=np.float64)
        self.b = np.zeros((feature_dim, 1), dtype=np.float64)
        self.A_inv = np.eye(feature_dim, dtype=np.float64)
        self.recompute_inv = True

    def get_action(self, state: np.ndarray, num_clients: int, k: int, context: Dict[str, Any] = None) -> List[int]:
        print("using Weighted Least Squares Thompson Sampling (WLS-TS)")
        context = context or {}
        active_indices = context.get("active_indices", list(range(num_clients)))
        k = min(k, len(active_indices))
        if k == 0 or state.shape[0] == 0:
            return []

        if state.ndim == 2 and state.shape[1] != self.feature_dim:
            self._init_dim(state.shape[1])

        if self.recompute_inv:
            self.A_inv = np.linalg.inv(self.A)
            self.recompute_inv = False

        hat_theta = (self.A_inv @ self.b).flatten()  # (feature_dim,)
        cov = (self.sigma ** 2) * self.A_inv  # (feature_dim, feature_dim)

        # Draw posterior sample tilde_theta ~ N(hat_theta, cov)
        try:
            tilde_theta = np.random.multivariate_normal(hat_theta, cov)
        except Exception:
            tilde_theta = hat_theta

        scores = np.full(num_clients, -np.inf, dtype=np.float64)
        for idx in active_indices:
            x_i = state[idx]  # (feature_dim,)
            scores[idx] = float(np.dot(tilde_theta, x_i))

        selected_indices = np.argsort(scores)[::-1][:k].tolist()
        print(f"[WLS-TS Agent] Selected top {len(selected_indices)} clients out of {len(active_indices)} active clients via Thompson Sampling.")
        return selected_indices

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, context: Dict[str, Any] = None):
        context = context or {}
        vector_rewards = context.get("vector_rewards", {})

        if state.ndim == 2 and state.shape[1] != self.feature_dim:
            self._init_dim(state.shape[1])

        # Apply exponential discounting factor gamma
        self.A = self.gamma * self.A + (1.0 - self.gamma) * np.eye(self.feature_dim, dtype=np.float64)
        self.b = self.gamma * self.b

        for idx in range(len(state)):
            x_i = state[idx].reshape(-1, 1)
            r_i = vector_rewards.get(idx, reward if idx in action else 0.0)
            
            if idx in action or idx in vector_rewards:
                self.A += x_i @ x_i.T
                self.b += r_i * x_i
                self.recompute_inv = True

        print(f"[WLS-TS Agent] Updated WLS-TS posterior distribution parameters (gamma={self.gamma}).")

class DQNAgent(BaseRLAgent):
    """
    TensorFlow/Keras Deep Q-Network Agent for client selection.
    Supports both feature_dim = 8 (Standalone mode) and feature_dim = 15 (Hierarchical mode).
    """
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 32, lr: float = 0.001, gamma: float = 0.9, epsilon: float = 0.1):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = []
        self.max_buffer_size = 1000
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_counter = 0

    def _build_model(self) -> tf.keras.Model:
        model = models.Sequential([
            layers.Input(shape=(self.feature_dim,)),
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def get_action(self, state: np.ndarray, num_clients: int, k: int, context: Dict[str, Any] = None) -> List[int]:
        if state.ndim == 2 and state.shape[1] != self.feature_dim:
            self.feature_dim = state.shape[1]
            self.model = self._build_model()
            self.target_model = self._build_model()

        print(f"using TensorFlow DQNAgent (dim={self.feature_dim})")
        context = context or {}
        active_indices = context.get("active_indices", list(range(num_clients)))
        k = min(k, len(active_indices))
        if k == 0 or state.shape[0] == 0:
            return []

        # Predict Q-values for all clients in state
        q_values = self.model.predict(state, verbose=0).flatten()

        scores = np.full(num_clients, -np.inf, dtype=np.float64)
        for idx in active_indices:
            if np.random.rand() < self.epsilon:
                scores[idx] = np.random.rand()
            else:
                scores[idx] = float(q_values[idx])

        # Pick top k indices with highest scores among active clients (Action Masking)
        selected_indices = np.argsort(scores)[::-1][:k].tolist()
        print(f"[DQN Agent] Selected top {len(selected_indices)} clients out of {len(active_indices)} active clients.")
        return selected_indices

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, context: Dict[str, Any] = None):
        context = context or {}
        vector_rewards = context.get("vector_rewards", {})

        # Store transition features per client
        for idx in range(len(state)):
            x_i = state[idx]
            r_i = vector_rewards.get(idx, reward if idx in action else 0.0)
            x_next_i = next_state[idx] if idx < len(next_state) else x_i
            
            if len(self.replay_buffer) >= self.max_buffer_size:
                self.replay_buffer.pop(0)
            self.replay_buffer.append((x_i, r_i, x_next_i))

        # Train model using mini-batch from replay buffer
        batch_size = min(32, len(self.replay_buffer))
        if batch_size > 0:
            indices = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in indices]
            
            states_b = np.array([item[0] for item in batch], dtype=np.float32)
            rewards_b = np.array([item[1] for item in batch], dtype=np.float32)
            next_states_b = np.array([item[2] for item in batch], dtype=np.float32)

            next_q_target = self.target_model.predict(next_states_b, verbose=0).flatten()
            y_targets = rewards_b + self.gamma * next_q_target
            
            self.model.train_on_batch(states_b, y_targets)

        self.update_target_counter += 1
        if self.update_target_counter % 5 == 0:
            self.target_model.set_weights(self.model.get_weights())
            print("[DQN Agent] Updated target network weights.")


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

class MetaAggregatorAgent:
    """
    Level 1 Meta-Controller Agent for selecting the aggregation strategy.
    Supports both Discounted-LinUCB (D-LinUCB) and Weighted Least Squares Thompson Sampling (WLS-TS).
    Strategies: ["FedAvg", "qFedAvg", "FedFV", "FedAdam", "FedProx", "Krum", "SCAFFOLD"]
    Input State S^(1) in R^5: [round_progress, global_loss_delta, loss_variance, avg_system_latency, poison_alert_flag]
    """
    STRATEGIES = ["FedAvg", "qFedAvg", "FedFV", "FedAdam", "FedProx", "Krum", "SCAFFOLD"]

    def __init__(self, mode: str = "d-linucb", alpha: float = 1.0, gamma: float = 0.95, sigma: float = 0.25, feature_dim: int = 5):
        self.mode = mode.lower()
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.feature_dim = feature_dim
        self.num_actions = len(self.STRATEGIES)
        self.A = [np.eye(feature_dim, dtype=np.float64) for _ in range(self.num_actions)]
        self.b = [np.zeros((feature_dim, 1), dtype=np.float64) for _ in range(self.num_actions)]
        self.last_action_idx = 0

    def select_strategy(self, global_state: np.ndarray) -> Tuple[int, str]:
        x = global_state.reshape(-1, 1)
        scores = np.zeros(self.num_actions, dtype=np.float64)

        if self.mode in ["wls-ts", "wlsts", "thompson"]:
            for k in range(self.num_actions):
                A_inv = np.linalg.inv(self.A[k])
                hat_theta = (A_inv @ self.b[k]).flatten()
                cov = (self.sigma ** 2) * A_inv
                try:
                    tilde_theta = np.random.multivariate_normal(hat_theta, cov)
                except Exception:
                    tilde_theta = hat_theta
                scores[k] = float(np.dot(tilde_theta, global_state))
            best_idx = int(np.argmax(scores))
            print(f"[Meta-Controller (WLS-TS)] Selected Aggregation Strategy: {self.STRATEGIES[best_idx]} (index {best_idx}) via Thompson Sampling")
        else:
            for k in range(self.num_actions):
                A_inv = np.linalg.inv(self.A[k])
                theta_k = A_inv @ self.b[k]
                exp_reward = float((theta_k.T @ x).item())
                uncert = float(np.sqrt((x.T @ A_inv @ x).item()))
                scores[k] = exp_reward + self.alpha * uncert

            # Random tie-breaking among maximum score candidates
            max_score = np.max(scores)
            candidates = np.where(np.isclose(scores, max_score, atol=1e-5))[0]
            best_idx = int(np.random.choice(candidates))
            print(f"[Meta-Controller (D-LinUCB)] Selected Aggregation Strategy: {self.STRATEGIES[best_idx]} (index {best_idx})")

        self.last_action_idx = best_idx
        strategy_name = self.STRATEGIES[best_idx]
        return best_idx, strategy_name

    def update(self, global_state: np.ndarray, action_idx: int, reward: float):
        x = global_state.reshape(-1, 1)
        # Apply exponential discounting factor gamma to past observations
        self.A[action_idx] = self.gamma * self.A[action_idx] + (1.0 - self.gamma) * np.eye(self.feature_dim, dtype=np.float64) + (x @ x.T)
        self.b[action_idx] = self.gamma * self.b[action_idx] + reward * x
        print(f"[Meta-Controller ({self.mode.upper()})] Updated weights for strategy '{self.STRATEGIES[action_idx]}' (gamma={self.gamma}).")

class HierarchicalFLSelector(ClientSelector):
    """
    2-Level Hierarchical RL Selector:
    Level 1: MetaAggregatorAgent selects 1 of 7 Aggregation Strategies.
    Level 2: LinUCBAgent selects Top-K Clients conditioned on chosen Aggregation Strategy.
    """
    def __init__(self, meta_agent: MetaAggregatorAgent = None, sub_agent: LinUCBAgent = None, env: Any = None):
        self.meta_agent = meta_agent or MetaAggregatorAgent()
        self.sub_agent = sub_agent or LinUCBAgent()
        self.env = env
        
        self.last_global_state = None
        self.last_agg_idx = 0
        self.last_chosen_agg = "FedAvg"
        self.last_client_state = None
        self.last_action = None
        self.last_client_ids = []

        self.client_staleness: Dict[str, int] = {}
        self.client_ema_latency: Dict[str, float] = {}
        self.client_ema_energy: Dict[str, float] = {}
        self.client_has_telemetry: Dict[str, float] = {}

    def select_clients(self, client_ids: List[str], k: int, context: Dict[str, Any] = None) -> List[str]:
        if not client_ids:
            return []
        
        context = context or {}
        active_clients = context.get("active_clients", client_ids)
        active_indices = [i for i, cid in enumerate(client_ids) if cid in active_clients]
        
        # Step 1: Level 1 Meta-Controller selects Aggregation Strategy
        global_state = self._build_global_state(context)
        self.last_global_state = global_state
        agg_idx, chosen_agg = self.meta_agent.select_strategy(global_state)
        self.last_agg_idx = agg_idx
        self.last_chosen_agg = chosen_agg
        context["chosen_aggregation"] = chosen_agg

        # Step 2: Level 2 Sub-Controller selects Top-K Clients conditioned on chosen_agg
        context["active_indices"] = active_indices
        context["env"] = self.env
        
        client_state = self._build_conditioned_client_state(client_ids, agg_idx, context)
        self.last_client_state = client_state
        self.last_client_ids = client_ids

        selected_indices = self.sub_agent.get_action(client_state, len(client_ids), k, context=context)
        self.last_action = selected_indices

        # Update staleness counters
        selected_ids = [client_ids[idx] for idx in selected_indices]
        for cid in client_ids:
            if cid in selected_ids:
                self.client_staleness[cid] = 0
            else:
                self.client_staleness[cid] = self.client_staleness.get(cid, 0) + 1

        return selected_ids

    def _build_global_state(self, context: Dict[str, Any]) -> np.ndarray:
        current_r = context.get("round", 1)
        total_r = max(1, current_r + context.get("rounds_left", 10))
        progress = float(current_r / total_r)
        
        global_loss_delta = float(context.get("global_loss_delta", 0.0))
        client_losses = list(context.get("client_losses", {}).values())
        loss_var = float(np.var(client_losses)) if len(client_losses) > 1 else 0.0
        
        avg_lat = float(np.mean(list(self.client_ema_latency.values()))) if self.client_ema_latency else 0.0
        poison_alert = float(context.get("poison_alert_flag", 0.0))
        
        return np.array([progress, global_loss_delta, loss_var, avg_lat, poison_alert], dtype=np.float32)

    def _build_conditioned_client_state(self, client_ids: List[str], agg_idx: int, context: Dict[str, Any]) -> np.ndarray:
        client_losses = context.get("client_losses", {})
        avg_loss = np.mean(list(client_losses.values())) if client_losses else 1.0
        
        # 7-dimensional one-hot vector for aggregation strategy
        one_hot_agg = [0.0] * len(MetaAggregatorAgent.STRATEGIES)
        if 0 <= agg_idx < len(one_hot_agg):
            one_hot_agg[agg_idx] = 1.0

        state_list = []
        for i, cid in enumerate(client_ids):
            num_id = context.get("client_id_map", {}).get(cid, i)
            profile = self.env.profiles.get(num_id, {"cpu_frequency": 2.0e9})
            
            samples = context.get("client_samples", {}).get(cid, 1000) / 1000.0
            last_loss = float(client_losses.get(cid, 1.0))
            loss_staleness = float(avg_loss - last_loss)
            
            # Telemetry metrics (default to 0.0 if not available per AGENTS.md rule)
            lat = float(self.client_ema_latency.get(cid, 0.0))
            eng = float(self.client_ema_energy.get(cid, 0.0))
            has_telemetry = float(self.client_has_telemetry.get(cid, 0.0))
            
            staleness = float(self.client_staleness.get(cid, 0))
            bias = 1.0
            
            # 8 base features + 7 one-hot features = 15 total features
            row = [
                last_loss, loss_staleness, lat, eng, has_telemetry, staleness, samples, bias
            ] + one_hot_agg
            state_list.append(row)
            
        return np.array(state_list, dtype=np.float32)

    def update_policy(self, round_summary: Dict[str, Any]):
        if self.last_client_state is None or self.last_action is None:
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
            
            alpha = 0.3
            self.client_ema_latency[cid] = (1 - alpha) * self.client_ema_latency.get(cid, 0.0) + alpha * cost_dict["t_total"]
            self.client_ema_energy[cid] = (1 - alpha) * self.client_ema_energy.get(cid, 0.0) + alpha * cost_dict["E_total"]
            self.client_has_telemetry[cid] = 1.0

        if hasattr(self.env, "calculate_vector_rewards"):
            c_rewards, scalar_reward = self.env.calculate_vector_rewards(
                self.last_client_ids, selected_ids, selected_metrics, 
                global_loss_delta, client_losses, self.client_staleness
            )
            vector_rewards = {i: c_rewards[cid] for i, cid in enumerate(self.last_client_ids) if cid in c_rewards}
        else:
            scalar_reward = self.env.calculate_reward(
                {client_id_map.get(cid, 0): m for cid, m in selected_metrics.items()},
                global_loss_delta, local_losses
            )
            vector_rewards = {}

        print(f"[RL Environment] Round {round_summary.get('round', 1)} Stats:")
        print(f"  - Chosen Aggregation Strategy: {self.last_chosen_agg}")
        print(f"  - Delta Global Loss: {global_loss_delta:.4f}")
        print(f"  - Calculated Reward: {scalar_reward:.4f}")

        # Update Level 1 Meta-Controller Policy
        if self.last_global_state is not None:
            self.meta_agent.update(self.last_global_state, self.last_agg_idx, scalar_reward)

        # Update Level 2 Sub-Controller Policy
        next_context = {
            "round": round_summary.get("round", 1),
            "rounds_left": round_summary.get("rounds_left", 0),
            "client_id_map": client_id_map,
            "client_samples": client_samples,
            "client_losses": client_losses,
            "active_clients": active_clients,
            "vector_rewards": vector_rewards
        }
        next_state = self._build_conditioned_client_state(self.last_client_ids, self.last_agg_idx, next_context)
        
        self.sub_agent.update(self.last_client_state, self.last_action, scalar_reward, next_state, context=next_context)


def get_selector_by_name(name: str, **kwargs) -> ClientSelector:
    """
    Factory function returning an instance of the requested ClientSelector strategy.
    Supported names: 'random' (default), 'linucb' / 'd-linucb', 'wls-ts' / 'thompson', 'dqn', 'hierarchical'
    """
    name_lower = (name or "random").lower()
    if name_lower == "random":
        return RandomClientSelector()
    elif name_lower in ["linucb", "d-linucb"]:
        env = kwargs.get("env")
        feature_dim = kwargs.get("feature_dim", 8)
        agent = LinUCBAgent(feature_dim=feature_dim)
        return RLClientSelector(agent, env)
    elif name_lower in ["wls-ts", "wlsts", "thompson"]:
        env = kwargs.get("env")
        feature_dim = kwargs.get("feature_dim", 8)
        agent = WLSTSAgent(feature_dim=feature_dim)
        return RLClientSelector(agent, env)
    elif name_lower == "dqn":
        env = kwargs.get("env")
        agent = DQNAgent()
        return RLClientSelector(agent, env)
    elif name_lower == "hierarchical":
        env = kwargs.get("env")
        meta_agent = kwargs.get("meta_agent") or MetaAggregatorAgent()
        sub_agent = kwargs.get("sub_agent") or LinUCBAgent()
        return HierarchicalFLSelector(meta_agent=meta_agent, sub_agent=sub_agent, env=env)
    else:
        print(f"[Selector Warning] Unknown selector strategy '{name}'. Defaulting to RandomClientSelector.")
        return RandomClientSelector()