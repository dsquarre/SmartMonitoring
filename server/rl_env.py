import numpy as np

class FederatedEnv:
    def __init__(self, client_profiles, model_size_bits=10_000_000, kappa=1e-27, cycles_per_sample=1e6):
        # client_profiles: Dict mapping numeric ID (int) -> Profile dict
        self.profiles = client_profiles
        self.model_size_bits = model_size_bits
        self.kappa = kappa
        self.cycles_per_sample = cycles_per_sample

    def compute_client_cost(self, numeric_id, samples, measured_roundtrip=None):
        profile = self.profiles[numeric_id]
        f = profile["cpu_frequency"]
        P_tx = profile["tx_power"]
        
        # 1. Local Training Latency and Energy (BEFL Model)
        epochs = 1
        t_train = (epochs * self.cycles_per_sample * samples) / f
        E_train = self.kappa * epochs * self.cycles_per_sample * samples * (f ** 2)

        # 2. Transmission Latency and Energy (BEFL Model)
        if measured_roundtrip is not None:
            # Derive transmission latency from actual WebSocket roundtrip time
            t_trans = max(0.001, measured_roundtrip - t_train)
        else:
            # Fallback to simulated channel upload rate
            t_trans = self.model_size_bits / profile["r_trans"]
            
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

        # Multi-objective Reward formulation
        reward = (w_perf * global_loss_delta) - (w_local * avg_local_loss) - (w_lat * max_latency) - (w_eng * total_energy) - (w_fair * loss_variance)
        return reward
