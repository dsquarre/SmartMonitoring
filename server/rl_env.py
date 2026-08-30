import numpy as np

class RunningNormalizer:
    """
    Online Running Mean & Variance Normalizer using Welford's algorithm / Exponential Standardization.
    Updates running mean (mu) and running variance (var) in O(1) memory and time.
    """
    def __init__(self, alpha: float = 0.05, epsilon: float = 1e-6):
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, val: float):
        val = float(val)
        if not self.initialized:
            self.mean = val
            self.var = 1.0
            self.initialized = True
        else:
            diff = val - self.mean
            self.mean += self.alpha * diff
            self.var = (1.0 - self.alpha) * self.var + self.alpha * (diff ** 2)

    def normalize(self, val: float) -> float:
        val = float(val)
        self.update(val)
        std = np.sqrt(max(self.var, self.epsilon))
        return (val - self.mean) / std

class FederatedEnv:
    def __init__(self, client_profiles, model_size_bits=10_000_000, kappa=1e-27, cycles_per_sample=1e6):
        # client_profiles: Dict mapping numeric ID (int) -> Profile dict
        self.profiles = client_profiles
        self.model_size_bits = model_size_bits
        self.kappa = kappa
        self.cycles_per_sample = cycles_per_sample
        
        # Online running normalizers for latency, energy, and local loss
        self.lat_normalizer = RunningNormalizer(alpha=0.05)
        self.eng_normalizer = RunningNormalizer(alpha=0.05)
        self.loss_normalizer = RunningNormalizer(alpha=0.05)

    def compute_client_cost(self, numeric_id, samples, actual_comp_latency=None, actual_measured_energy=None, measured_roundtrip=None):
        profile = self.profiles[numeric_id]
        P_tx = profile.get("tx_power", 0.2)
        f = profile.get("cpu_frequency", 2.0e9)
        
        # 1. Local Training Latency and Energy (default to 0.0 if not available)
        t_train = actual_comp_latency if actual_comp_latency is not None else 0.0
        E_train = actual_measured_energy if actual_measured_energy is not None else 0.0

        # 2. Transmission Latency and Energy
        if measured_roundtrip is not None:
            # Derive transmission latency from actual WebSocket roundtrip time minus local training latency
            t_trans = max(0.001, measured_roundtrip - t_train)
        else:
            # Fallback to simulated channel upload rate
            t_trans = self.model_size_bits / profile.get("r_trans", 15e6)
            
        E_trans = P_tx * t_trans

        return {
            "t_train": t_train,
            "t_trans": t_trans,
            "t_total": t_train + t_trans,
            "E_train": E_train,
            "E_trans": E_trans,
            "E_total": E_train + E_trans
        }

    def calculate_reward(self, selected_metrics, global_loss_delta, local_losses, 
                         w_perf=10.0, w_local=1.0, w_lat=0.1, w_eng=1.0, w_fair=0.5):
        # Latency is determined by the slowest client (straggler)
        max_latency = max(m["t_total"] for m in selected_metrics.values()) if selected_metrics else 0.0
        
        # Total energy is sum across selected clients
        total_energy = sum(m["E_total"] for m in selected_metrics.values()) if selected_metrics else 0.0
        
        # Performance/loss statistics
        avg_local_loss = np.mean(local_losses) if local_losses else 1.0
        loss_variance = np.var(local_losses) if len(local_losses) > 1 else 0.0

        # Standardize metric terms dynamically using online running Welford statistics
        norm_lat = self.lat_normalizer.normalize(max_latency)
        norm_eng = self.eng_normalizer.normalize(total_energy)
        norm_loss = self.loss_normalizer.normalize(avg_local_loss)

        # Multi-objective Reward formulation
        reward = (w_perf * global_loss_delta) - (w_local * norm_loss) - (w_lat * norm_lat) - (w_eng * norm_eng) - (w_fair * loss_variance)
        return reward

    def calculate_vector_rewards(self, client_ids, selected_ids, selected_metrics, global_loss_delta, 
                                 client_losses, staleness_dict=None,
                                 w_perf=10.0, w_local=1.0, w_lat=0.1, w_eng=1.0, w_stale=0.05):
        """
        Calculates per-client individual reward vector and global composite reward scalar.
        
        Returns:
            client_rewards: Dict[str, float] mapping client_id -> individual reward.
            scalar_reward: float total reward scalar.
        """
        staleness_dict = staleness_dict or {}
        client_rewards = {}
        
        for cid in client_ids:
            if cid in selected_ids and cid in selected_metrics:
                m = selected_metrics[cid]
                c_loss = client_losses.get(cid, 1.0)
                
                # Standardize individual client metrics dynamically
                norm_loss = self.loss_normalizer.normalize(c_loss)
                norm_lat = self.lat_normalizer.normalize(m.get("t_total", 0.0))
                norm_eng = self.eng_normalizer.normalize(m.get("E_total", 0.0))

                # Selected client reward using standardized metrics
                r_i = (w_perf * global_loss_delta) - (w_local * norm_loss) - (w_lat * norm_lat) - (w_eng * norm_eng)
            else:
                # Unselected client: penalty proportional to staleness to discourage starvation
                stale_rounds = staleness_dict.get(cid, 0)
                r_i = - (w_stale * stale_rounds)
            client_rewards[cid] = float(r_i)

        scalar_reward = sum(client_rewards.values())
        return client_rewards, scalar_reward


