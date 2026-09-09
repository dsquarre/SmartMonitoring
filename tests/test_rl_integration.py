import sys
import os

# Append server directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from rl_env import FederatedEnv

def test_rl_cost_calculation():
    # Setup test profile
    profiles = {
        0: {
            "cpu_frequency": 2.0e9,
            "tx_power": 0.2,
            "r_trans": 15e6
        }
    }
    
    # Initialize environment
    env = FederatedEnv(profiles, model_size_bits=10_000_000)
    
    # Test values
    numeric_id = 0
    samples = 1000
    actual_comp_latency = 3.5  # seconds
    actual_measured_energy = 52.5  # Joules
    measured_roundtrip = 4.0  # seconds
    
    # Run cost calculation
    cost = env.compute_client_cost(
        numeric_id=numeric_id,
        samples=samples,
        actual_comp_latency=actual_comp_latency,
        actual_measured_energy=actual_measured_energy,
        measured_roundtrip=measured_roundtrip
    )
    
    print("Computed Cost Profile:")
    for key, value in cost.items():
        print(f"  - {key}: {value}")
        
    # Assertions
    assert cost["t_train"] == actual_comp_latency, f"Expected t_train={actual_comp_latency}, got {cost['t_train']}"
    assert cost["E_train"] == actual_measured_energy, f"Expected E_train={actual_measured_energy}, got {cost['E_train']}"
    
    # t_trans = measured_roundtrip - actual_comp_latency = 4.0 - 3.5 = 0.5
    assert abs(cost["t_trans"] - 0.5) < 1e-9, f"Expected t_trans=0.5, got {cost['t_trans']}"
    
    # E_trans = tx_power * t_trans = 0.2 * 0.5 = 0.1
    assert abs(cost["E_trans"] - 0.1) < 1e-9, f"Expected E_trans=0.1, got {cost['E_trans']}"
    
    # Totals
    assert abs(cost["t_total"] - 4.0) < 1e-9, f"Expected t_total=4.0, got {cost['t_total']}"
    assert abs(cost["E_total"] - 52.6) < 1e-9, f"Expected E_total=52.6, got {cost['E_total']}"
    
    print("\nINTEGRATION TEST PASSED!")

if __name__ == "__main__":
    test_rl_cost_calculation()
