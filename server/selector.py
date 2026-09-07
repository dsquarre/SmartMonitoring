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
    def __init__(self, alpha: float = 1.0, gamma: float = 0.95, feature_dim: int = 7):
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
    def __init__(self, gamma: float = 0.95, sigma: float = 0.25, feature_dim: int = 7):
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
    Supports both feature_dim = 7 (Standalone mode) and feature_dim = 14 (Hierarchical mode).
    """
    def __init__(self, feature_dim: int = 7, hidden_dim: int = 32, lr: float = 0.001, gamma: float = 0.9, epsilon: float = 0.1):
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


def compute_selection_diversity(selection_history: List[List[str]], client_ids: List[str], window_size: int = 10) -> float:
    """
    Computes normalized Shannon entropy diversity_t = H / log(N) over trailing window W of rounds.
    1.0 means policy cycles through pool evenly; near 0.0 means hammering same clients.
    """
    N = len(client_ids)
    if N <= 1 or not selection_history:
        return 1.0

    window = selection_history[-window_size:]
    total_selections = sum(len(s) for s in window)
    if total_selections == 0:
        return 1.0

    counts = {cid: 0 for cid in client_ids}
    for round_selected in window:
        for cid in round_selected:
            if cid in counts:
                counts[cid] += 1

    probs = [counts[cid] / float(total_selections) for cid in client_ids]
    H = 0.0
    for p in probs:
        if p > 0:
            H -= p * np.log(p)

    max_H = np.log(N)
    if max_H <= 0:
        return 1.0
    diversity = float(H / max_H)
    return float(np.clip(diversity, 0.0, 1.0))


def compute_mad_anomaly_fraction(client_grad_sims: Dict[str, float], selected_ids: List[str], kappa: float = 2.5) -> float:
    """
    Computes robust MAD-based anomaly fraction p / |S_t| across responding clients this round.
    m = median(c_i)
    MAD = median(|c_i - m|)
    flag_i = 1[c_i < m - kappa * MAD]
    anomaly_fraction = sum(flag_i) / |S_t|
    """
    if not selected_ids:
        return 0.0

    c_vals = [float(client_grad_sims[cid]) for cid in selected_ids if cid in client_grad_sims]
    S_t_count = len(c_vals)
    if S_t_count == 0:
        return 0.0

    c_arr = np.array(c_vals, dtype=np.float64)
    m = float(np.median(c_arr))
    mad = float(np.median(np.abs(c_arr - m)))

    if mad <= 1e-8:
        flags = (c_arr < (m - 1e-5)).astype(np.float64)
    else:
        threshold = m - kappa * mad
        flags = (c_arr < threshold).astype(np.float64)

    p = float(np.sum(flags))
    return float(p / S_t_count)


def build_base_client_features(
    client_ids: List[str],
    context: Dict[str, Any],
    client_ema_loss: Dict[str, float],
    ema_global_loss: float,
    client_ema_latency: Dict[str, float],
    client_ema_energy: Dict[str, float],
    client_staleness: Dict[str, int],
    client_dropped: Dict[str, float],
    client_grad_sim: Dict[str, float],
    selection_history: List[List[str]],
    window_size: int = 10
) -> np.ndarray:
    """
    Builds N x 7 feature matrix for clients:
    [loss_diff, EMA_latency, EMA_energy, stale_count, drop_flag, grad_sim, entropy_diversity]
    Defaults unobserved loss, latency, energy, staleness to float('inf') / np.inf.
    """
    diversity_t = compute_selection_diversity(selection_history, client_ids, window_size)
    client_losses = context.get("client_losses", {})

    state_list = []
    for cid in client_ids:
        c_loss_ema = client_ema_loss.get(cid, float(client_losses.get(cid, np.inf)))
        if np.isinf(c_loss_ema) or np.isinf(ema_global_loss):
            loss_diff = np.inf
        else:
            loss_diff = float(c_loss_ema - ema_global_loss)

        lat = float(client_ema_latency.get(cid, np.inf))
        eng = float(client_ema_energy.get(cid, np.inf))
        staleness = float(client_staleness.get(cid, np.inf))
        drop_flag = float(client_dropped.get(cid, 0.0))
        grad_sim = float(client_grad_sim.get(cid, 0.0))
        entropy_feat = float(diversity_t)

        state_list.append([
            loss_diff,
            lat,
            eng,
            staleness,
            drop_flag,
            grad_sim,
            entropy_feat
        ])
    raw_state = np.array(state_list, dtype=np.float32)
    return np.nan_to_num(raw_state, nan=0.0, posinf=1e6, neginf=-1e6)


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
        self.client_has_telemetry: Dict[str, float] = {}
        self.client_ema_loss: Dict[str, float] = {}
        self.ema_global_loss: float = np.inf
        self.client_dropped: Dict[str, float] = {}
        self.client_grad_sim: Dict[str, float] = {}
        self.selection_history: List[List[str]] = []
        self.window_size: int = 10

    def select_clients(self, client_ids: List[str], k: int, context: Dict[str, Any] = None) -> List[str]:
        if not client_ids:
            return []

        context = context or {}
        active_clients = context.get("active_clients", client_ids)
        active_indices = [i for i, cid in enumerate(client_ids) if cid in active_clients]

        context["active_indices"] = active_indices
        context["env"] = self.env

        state = self._build_state(client_ids, context)
        self.last_state = state
        self.last_client_ids = client_ids

        selected_indices = self.agent.get_action(state, len(client_ids), k, context=context)
        self.last_action = selected_indices

        selected_ids = [client_ids[idx] for idx in selected_indices]
        for cid in client_ids:
            if cid in selected_ids:
                self.client_staleness[cid] = 0
            else:
                self.client_staleness[cid] = self.client_staleness.get(cid, 0) + 1

        return selected_ids

    def _build_state(self, client_ids: List[str], context: Dict[str, Any]) -> np.ndarray:
        return build_base_client_features(
            client_ids, context,
            self.client_ema_loss, self.ema_global_loss,
            self.client_ema_latency, self.client_ema_energy,
            self.client_staleness, self.client_dropped,
            self.client_grad_sim, self.selection_history,
            window_size=self.window_size
        )

    def update_policy(self, round_summary: Dict[str, Any]):
        if self.last_state is None or self.last_action is None:
            return

        selected_ids = round_summary.get("selected_ids", [])
        self.selection_history.append(selected_ids)
        if len(self.selection_history) > self.window_size * 2:
            self.selection_history = self.selection_history[-self.window_size:]

        client_id_map = round_summary.get("client_id_map", {})
        client_samples = round_summary.get("client_samples", {})
        client_losses = round_summary.get("client_losses", {})
        global_loss_delta = round_summary.get("global_loss_delta", 0.0)
        local_losses = round_summary.get("local_losses", [])
        active_clients = round_summary.get("active_clients", self.last_client_ids)
        roundtrips = round_summary.get("client_roundtrips", {})
        latencies = round_summary.get("client_latencies", {})
        energies = round_summary.get("client_energies", {})
        dropped_clients = set(round_summary.get("dropped_clients", []))
        client_grad_sims = round_summary.get("client_grad_sims", {})

        alpha = 0.3
        if client_losses:
            for cid, c_loss in client_losses.items():
                c_loss = float(c_loss)
                if cid in self.client_ema_loss and not np.isinf(self.client_ema_loss[cid]):
                    self.client_ema_loss[cid] = (1.0 - alpha) * self.client_ema_loss[cid] + alpha * c_loss
                else:
                    self.client_ema_loss[cid] = c_loss
            curr_global = float(np.mean(list(client_losses.values())))
            if np.isinf(self.ema_global_loss):
                self.ema_global_loss = curr_global
            else:
                self.ema_global_loss = (1.0 - alpha) * self.ema_global_loss + alpha * curr_global

        for cid in self.last_client_ids:
            self.client_dropped[cid] = 1.0 if cid in dropped_clients else 0.0

        for cid, sim in client_grad_sims.items():
            self.client_grad_sim[cid] = float(sim)

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

            curr_lat = cost_dict["t_total"]
            curr_eng = cost_dict["E_total"]
            if np.isinf(self.client_ema_latency.get(cid, np.inf)):
                self.client_ema_latency[cid] = curr_lat
            else:
                self.client_ema_latency[cid] = (1 - alpha) * self.client_ema_latency[cid] + alpha * curr_lat

            if np.isinf(self.client_ema_energy.get(cid, np.inf)):
                self.client_ema_energy[cid] = curr_eng
            else:
                self.client_ema_energy[cid] = (1 - alpha) * self.client_ema_energy[cid] + alpha * curr_eng

            self.client_has_telemetry[cid] = 1.0

        if hasattr(self.env, "calculate_vector_rewards"):
            c_rewards, reward = self.env.calculate_vector_rewards(
                self.last_client_ids, selected_ids, selected_metrics,
                global_loss_delta, client_losses, self.client_staleness
            )
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
    Input State G_t in R^20: [13 base numerical features + 7 one-hot last action]
    """
    STRATEGIES = ["FedAvg", "qFedAvg", "FedFV", "FedAdam", "FedProx", "Krum", "SCAFFOLD"]

    def __init__(self, mode: str = "d-linucb", alpha: float = 1.0, gamma: float = 0.95, sigma: float = 0.25, feature_dim: int = 20):
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
        clean_state = np.nan_to_num(global_state, nan=0.0, posinf=1e6, neginf=-1e6)
        x = clean_state.reshape(-1, 1)
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
                scores[k] = float(np.dot(tilde_theta, clean_state))
            best_idx = int(np.argmax(scores))
            print(f"[Meta-Controller (WLS-TS)] Selected Aggregation Strategy: {self.STRATEGIES[best_idx]} (index {best_idx}) via Thompson Sampling")
        else:
            for k in range(self.num_actions):
                A_inv = np.linalg.inv(self.A[k])
                theta_k = A_inv @ self.b[k]
                exp_reward = float((theta_k.T @ x).item())
                uncert = float(np.sqrt((x.T @ A_inv @ x).item()))
                scores[k] = exp_reward + self.alpha * uncert

            max_score = np.max(scores)
            candidates = np.where(np.isclose(scores, max_score, atol=1e-5))[0]
            best_idx = int(np.random.choice(candidates))
            print(f"[Meta-Controller (D-LinUCB)] Selected Aggregation Strategy: {self.STRATEGIES[best_idx]} (index {best_idx})")

        self.last_action_idx = best_idx
        strategy_name = self.STRATEGIES[best_idx]
        return best_idx, strategy_name

    def update(self, global_state: np.ndarray, action_idx: int, reward: float):
        clean_state = np.nan_to_num(global_state, nan=0.0, posinf=1e6, neginf=-1e6)
        x = clean_state.reshape(-1, 1)
        self.A[action_idx] = self.gamma * self.A[action_idx] + (1.0 - self.gamma) * np.eye(self.feature_dim, dtype=np.float64) + (x @ x.T)
        self.b[action_idx] = self.gamma * self.b[action_idx] + reward * x
        print(f"[Meta-Controller ({self.mode.upper()})] Updated weights for strategy '{self.STRATEGIES[action_idx]}' (gamma={self.gamma}).")


class HierarchicalFLSelector(ClientSelector):
    """
    2-Level Hierarchical RL Selector:
    Level 1: MetaAggregatorAgent selects 1 of 7 Aggregation Strategies using 20-dim G_t state vector.
    Level 2: LinUCBAgent selects Top-K Clients conditioned on chosen Aggregation Strategy using 14-dim X_t state vector.
    """
    def __init__(self, meta_agent: MetaAggregatorAgent = None, sub_agent: LinUCBAgent = None, env: Any = None):
        self.meta_agent = meta_agent or MetaAggregatorAgent(feature_dim=20)
        self.sub_agent = sub_agent or LinUCBAgent(feature_dim=14)
        self.env = env

        self.last_global_state = None
        self.last_agg_idx = 0
        self.last_chosen_agg = "FedAvg"
        self.last_client_state = None
        self.last_action = None
        self.last_client_ids = []
        self.last_selected_ids = []

        self.client_staleness: Dict[str, int] = {}
        self.client_ema_latency: Dict[str, float] = {}
        self.client_ema_energy: Dict[str, float] = {}
        self.client_has_telemetry: Dict[str, float] = {}
        self.client_ema_loss: Dict[str, float] = {}
        self.ema_global_loss: float = np.inf
        self.ema_global_acc: float = np.inf
        self.ema_global_lat: float = np.inf
        self.ema_global_eng: float = np.inf
        self.client_dropped: Dict[str, float] = {}
        self.client_grad_sim: Dict[str, float] = {}
        self.selection_history: List[List[str]] = []
        self.loss_delta_history: List[float] = []
        self.rounds_since_last_switch: int = 0
        self.window_size: int = 10

    def select_clients(self, client_ids: List[str], k: int, context: Dict[str, Any] = None) -> List[str]:
        if not client_ids:
            return []

        context = context or {}
        active_clients = context.get("active_clients", client_ids)
        active_indices = [i for i, cid in enumerate(client_ids) if cid in active_clients]

        global_state = self._build_global_state(context)
        self.last_global_state = global_state
        agg_idx, chosen_agg = self.meta_agent.select_strategy(global_state)

        if agg_idx == self.last_agg_idx:
            self.rounds_since_last_switch += 1
        else:
            self.rounds_since_last_switch = 0

        self.last_agg_idx = agg_idx
        self.last_chosen_agg = chosen_agg
        context["chosen_aggregation"] = chosen_agg

        context["active_indices"] = active_indices
        context["env"] = self.env

        client_state = self._build_conditioned_client_state(client_ids, agg_idx, context)
        self.last_client_state = client_state
        self.last_client_ids = client_ids

        selected_indices = self.sub_agent.get_action(client_state, len(client_ids), k, context=context)
        self.last_action = selected_indices

        selected_ids = [client_ids[idx] for idx in selected_indices]
        self.last_selected_ids = selected_ids
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

        ema_loss = float(self.ema_global_loss)
        ema_acc = float(self.ema_global_acc)
        ema_lat = float(self.ema_global_lat)
        ema_eng = float(self.ema_global_eng)

        dropped_list = context.get("dropped_clients", [])
        k_sel = max(1, context.get("select_k", len(self.last_selected_ids) or 1))
        pool_dropout_rate = float(len(dropped_list) / k_sel)

        delta_trend = float(np.mean(self.loss_delta_history[-5:])) if self.loss_delta_history else 0.0

        c_accs = list(context.get("client_accuracies", {}).values())
        var_acc = float(np.var(c_accs)) if len(c_accs) > 1 else 0.0

        c_losses = list(context.get("client_losses", {}).values())
        var_loss = float(np.var(c_losses)) if len(c_losses) > 1 else 0.0
        disparity_loss = float(np.max(c_losses) - np.min(c_losses)) if c_losses else 0.0

        client_ids = context.get("active_clients", self.last_client_ids)
        diversity_feat = float(compute_selection_diversity(self.selection_history, client_ids, self.window_size))

        anomaly_frac = float(compute_mad_anomaly_fraction(self.client_grad_sim, self.last_selected_ids, kappa=2.5))

        rounds_switch = float(self.rounds_since_last_switch)

        one_hot_last_action = [0.0] * len(MetaAggregatorAgent.STRATEGIES)
        if 0 <= self.last_agg_idx < len(one_hot_last_action):
            one_hot_last_action[self.last_agg_idx] = 1.0

        global_features = [
            ema_loss,
            ema_acc,
            ema_lat,
            ema_eng,
            progress,
            pool_dropout_rate,
            delta_trend,
            var_acc,
            var_loss,
            disparity_loss,
            diversity_feat,
            anomaly_frac,
            rounds_switch
        ] + one_hot_last_action

        raw_global = np.array(global_features, dtype=np.float32)
        return np.nan_to_num(raw_global, nan=0.0, posinf=1e6, neginf=-1e6)

    def _build_conditioned_client_state(self, client_ids: List[str], agg_idx: int, context: Dict[str, Any]) -> np.ndarray:
        base_state = build_base_client_features(
            client_ids, context,
            self.client_ema_loss, self.ema_global_loss,
            self.client_ema_latency, self.client_ema_energy,
            self.client_staleness, self.client_dropped,
            self.client_grad_sim, self.selection_history,
            window_size=self.window_size
        )

        one_hot_agg = [0.0] * len(MetaAggregatorAgent.STRATEGIES)
        if 0 <= agg_idx < len(one_hot_agg):
            one_hot_agg[agg_idx] = 1.0

        one_hot_matrix = np.tile(one_hot_agg, (len(client_ids), 1))
        conditioned_state = np.hstack([base_state, one_hot_matrix])
        return conditioned_state.astype(np.float32)

    def update_policy(self, round_summary: Dict[str, Any]):
        if self.last_client_state is None or self.last_action is None:
            return

        selected_ids = round_summary.get("selected_ids", [])
        self.selection_history.append(selected_ids)
        if len(self.selection_history) > self.window_size * 2:
            self.selection_history = self.selection_history[-self.window_size:]

        client_id_map = round_summary.get("client_id_map", {})
        client_samples = round_summary.get("client_samples", {})
        client_losses = round_summary.get("client_losses", {})
        global_loss_delta = round_summary.get("global_loss_delta", 0.0)
        local_losses = round_summary.get("local_losses", [])
        active_clients = round_summary.get("active_clients", self.last_client_ids)
        roundtrips = round_summary.get("client_roundtrips", {})
        latencies = round_summary.get("client_latencies", {})
        energies = round_summary.get("client_energies", {})
        dropped_clients = set(round_summary.get("dropped_clients", []))
        client_grad_sims = round_summary.get("client_grad_sims", {})

        self.loss_delta_history.append(float(global_loss_delta))
        if len(self.loss_delta_history) > 50:
            self.loss_delta_history = self.loss_delta_history[-20:]

        alpha = 0.3
        if client_losses:
            for cid, c_loss in client_losses.items():
                c_loss = float(c_loss)
                if cid in self.client_ema_loss and not np.isinf(self.client_ema_loss[cid]):
                    self.client_ema_loss[cid] = (1.0 - alpha) * self.client_ema_loss[cid] + alpha * c_loss
                else:
                    self.client_ema_loss[cid] = c_loss
            curr_global = float(np.mean(list(client_losses.values())))
            if np.isinf(self.ema_global_loss):
                self.ema_global_loss = curr_global
            else:
                self.ema_global_loss = (1.0 - alpha) * self.ema_global_loss + alpha * curr_global

        curr_acc = round_summary.get("global_accuracy", None)
        if curr_acc is not None:
            if np.isinf(self.ema_global_acc):
                self.ema_global_acc = float(curr_acc)
            else:
                self.ema_global_acc = (1 - alpha) * self.ema_global_acc + alpha * float(curr_acc)

        avg_lat = round_summary.get("avg_comp_latency", None)
        if avg_lat is not None:
            if np.isinf(self.ema_global_lat):
                self.ema_global_lat = float(avg_lat)
            else:
                self.ema_global_lat = (1 - alpha) * self.ema_global_lat + alpha * float(avg_lat)

        tot_eng = round_summary.get("total_round_energy", None)
        if tot_eng is not None:
            if np.isinf(self.ema_global_eng):
                self.ema_global_eng = float(tot_eng)
            else:
                self.ema_global_eng = (1 - alpha) * self.ema_global_eng + alpha * float(tot_eng)

        for cid in self.last_client_ids:
            self.client_dropped[cid] = 1.0 if cid in dropped_clients else 0.0

        for cid, sim in client_grad_sims.items():
            self.client_grad_sim[cid] = float(sim)

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

            curr_lat = cost_dict["t_total"]
            curr_eng = cost_dict["E_total"]
            if np.isinf(self.client_ema_latency.get(cid, np.inf)):
                self.client_ema_latency[cid] = curr_lat
            else:
                self.client_ema_latency[cid] = (1 - alpha) * self.client_ema_latency[cid] + alpha * curr_lat

            if np.isinf(self.client_ema_energy.get(cid, np.inf)):
                self.client_ema_energy[cid] = curr_eng
            else:
                self.client_ema_energy[cid] = (1 - alpha) * self.client_ema_energy[cid] + alpha * curr_eng

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

        if self.last_global_state is not None:
            self.meta_agent.update(self.last_global_state, self.last_agg_idx, scalar_reward)

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
        feature_dim = kwargs.get("feature_dim", 7)
        agent = LinUCBAgent(feature_dim=feature_dim)
        return RLClientSelector(agent, env)
    elif name_lower in ["wls-ts", "wlsts", "thompson"]:
        env = kwargs.get("env")
        feature_dim = kwargs.get("feature_dim", 7)
        agent = WLSTSAgent(feature_dim=feature_dim)
        return RLClientSelector(agent, env)
    elif name_lower == "dqn":
        env = kwargs.get("env")
        feature_dim = kwargs.get("feature_dim", 7)
        agent = DQNAgent(feature_dim=feature_dim)
        return RLClientSelector(agent, env)
    elif name_lower == "hierarchical":
        env = kwargs.get("env")
        meta_agent = kwargs.get("meta_agent") or MetaAggregatorAgent()
        sub_agent = kwargs.get("sub_agent") or LinUCBAgent(feature_dim=14)
        return HierarchicalFLSelector(meta_agent=meta_agent, sub_agent=sub_agent, env=env)
    else:
        print(f"[Selector Warning] Unknown selector strategy '{name}'. Defaulting to RandomClientSelector.")
        return RandomClientSelector()