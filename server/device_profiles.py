import numpy as np

# Realistic Edge Device Tiers for ECG / Wearable FL Experiments
EDGE_TIERS = {
    "Tier_1_Low_Power_Wearable": {  # e.g., Smart Band, low-power MCU
        "cpu_frequency": 0.8e9,       # 800 MHz
        "power_coeff": 1.2e-27,       # Power factor alpha
        "tx_power": 0.1,              # 100 mW transmission
        "r_trans": 5e6,               # 5 Mbps link
    },
    "Tier_2_Smartwatch": {           # e.g., Apple Watch, WearOS device
        "cpu_frequency": 1.5e9,       # 1.5 GHz
        "power_coeff": 1.0e-27,
        "tx_power": 0.2,              # 200 mW
        "r_trans": 15e6,              # 15 Mbps link
    },
    "Tier_3_Smartphone": {           # e.g., Mid-range Android / iPhone
        "cpu_frequency": 2.4e9,       # 2.4 GHz
        "power_coeff": 0.8e-27,
        "tx_power": 0.4,              # 400 mW
        "r_trans": 30e6,              # 30 Mbps link
    },
    "Tier_4_Edge_Hub": {              # e.g., Hospital Gateway / Raspberry Pi 4
        "cpu_frequency": 3.0e9,       # 3.0 GHz
        "power_coeff": 0.5e-27,
        "tx_power": 0.5,              # 500 mW
        "r_trans": 50e6,              # 50 Mbps link
    }
}

class DeviceProfileManager:
    """Manages reproducible client hardware heterogeneity assignment."""
    def __init__(self, num_clients=100, seed=42):
        np.random.seed(seed)
        self.tier_names = list(EDGE_TIERS.keys())
        self.client_profiles = {}
        
        # Assign clients randomly across the 4 hardware tiers
        for cid_idx in range(num_clients):
            chosen_tier = np.random.choice(self.tier_names, p=[0.3, 0.4, 0.2, 0.1])
            specs = EDGE_TIERS[chosen_tier].copy()
            specs["tier_name"] = chosen_tier
            self.client_profiles[cid_idx] = specs

    def compute_client_metrics(self, cid_idx, num_samples, epochs, model_size_bits=10_000_000, flops_per_sample=1_200_000):
        """Calculates realistic computation latency and total energy (comp + tx)."""
        specs = self.client_profiles[cid_idx]
        
        f_cpu = specs["cpu_frequency"]
        p_coeff = specs["power_coeff"]
        p_tx = specs["tx_power"]
        r_trans = specs["r_trans"]

        # 1. Computation Latency & Energy
        total_flops = flops_per_sample * num_samples * epochs
        t_comp = total_flops / f_cpu
        p_comp = p_coeff * (f_cpu ** 3)  # P = alpha * f^3
        e_comp = p_comp * t_comp

        # 2. Transmission Latency & Energy
        t_trans = model_size_bits / r_trans
        e_trans = p_tx * t_trans

        total_latency = t_comp + t_trans
        total_energy = e_comp + e_trans

        return {
            "comp_latency": float(t_comp),
            "trans_latency": float(t_trans),
            "total_latency": float(total_latency),
            "comp_energy": float(e_comp),
            "trans_energy": float(e_trans),
            "total_energy": float(total_energy),
            "tier": specs["tier_name"]
        }