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
        
        # Online running normalizers for latency, energy, local loss, global acc delta, global eng, global lat
        self.lat_normalizer = RunningNormalizer(alpha=0.05)
        self.eng_normalizer = RunningNormalizer(alpha=0.05)
        self.loss_normalizer = RunningNormalizer(alpha=0.05)
        self.acc_delta_normalizer = RunningNormalizer(alpha=0.05)
        self.global_eng_normalizer = RunningNormalizer(alpha=0.05)
        self.global_lat_normalizer = RunningNormalizer(alpha=0.05)

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

    def calculate_meta_reward(self, global_acc_delta: float, total_round_energy: float, round_latency: float) -> float:
        """
        Meta-Aggregator Reward r_t = 0.5 * (tanh(10 * z_W(ΔAcc_t) - z_W(E_t) - z_W(L_t)) + 1)
        """
        z_acc_delta = self.acc_delta_normalizer.normalize(global_acc_delta)
        z_eng = self.global_eng_normalizer.normalize(total_round_energy)
        z_lat = self.global_lat_normalizer.normalize(round_latency)

        raw_r = 10.0 * z_acc_delta - z_eng - z_lat
        return float(0.5 * (np.tanh(raw_r) + 1.0))

    def calculate_reward(self, selected_metrics, global_loss_delta, local_losses, 
                         w_perf=10.0, w_local=1.0, w_lat=0.1, w_eng=1.0, w_fair=0.5):
        max_latency = max(m["t_total"] for m in selected_metrics.values()) if selected_metrics else 0.0
        total_energy = sum(m["E_total"] for m in selected_metrics.values()) if selected_metrics else 0.0
        avg_local_loss = np.mean(local_losses) if local_losses else 1.0

        return self.calculate_meta_reward(global_loss_delta, total_energy, max_latency)

    def calculate_vector_rewards(self, client_ids, selected_ids, selected_metrics, global_loss_delta, 
                                 client_losses, staleness_dict=None,
                                 w_L=1.0, w_E=1.0, w_stale=0.05, global_acc_delta=None,
                                 total_round_energy=None, round_latency=None):
        """
        Calculates per-client sub-controller reward r_{t, i} = 0.5 * (tanh(z_W(loss) - w_L * z_W(lat) - w_E * z_W(eng)) + 1)
        and meta-aggregator reward r_t.
        """
        staleness_dict = staleness_dict or {}
        client_rewards = {}
        
        for cid in client_ids:
            if cid in selected_ids and cid in selected_metrics:
                m = selected_metrics[cid]
                c_loss = client_losses.get(cid, 1.0)
                
                z_loss = self.loss_normalizer.normalize(c_loss)
                z_lat = self.lat_normalizer.normalize(m.get("t_total", 0.0))
                z_eng = self.eng_normalizer.normalize(m.get("E_total", 0.0))

                raw_r = z_loss - (w_L * z_lat) - (w_E * z_eng)
                r_i = 0.5 * (np.tanh(raw_r) + 1.0)
            else:
                stale_rounds = staleness_dict.get(cid, 0)
                raw_r = - (w_stale * stale_rounds)
                r_i = 0.5 * (np.tanh(raw_r) + 1.0)
            client_rewards[cid] = float(r_i)

        scalar_reward = float(np.mean(list(client_rewards.values()))) if client_rewards else 0.5

        if global_acc_delta is not None and total_round_energy is not None and round_latency is not None:
            meta_reward = self.calculate_meta_reward(global_acc_delta, total_round_energy, round_latency)
        else:
            tot_eng = sum(m.get("E_total", 0.0) for m in selected_metrics.values()) if selected_metrics else 0.0
            max_lat = max(m.get("t_total", 0.0) for m in selected_metrics.values()) if selected_metrics else 0.0
            meta_reward = self.calculate_meta_reward(global_loss_delta, tot_eng, max_lat)

        return client_rewards, scalar_reward, meta_reward


