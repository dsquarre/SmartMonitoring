import sys
import os
import numpy as np

# Append server to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from model import Model
from aggregator import filter_poisoned_clients

def test_weights_poisoning_defense():
    print("Testing weights poisoning defense...")
    os.makedirs("tests/tmp", exist_ok=True)
    
    # 1. Create baseline model and save
    base_model = Model()
    global_path = "tests/tmp/global.keras"
    base_model.model.save(global_path)
    
    weights = base_model.model.get_weights()
    
    # 2. Create 3 healthy clients (small variance)
    h_paths = []
    for i in range(3):
        h_weights = [w + np.random.normal(0, 0.01, size=w.shape) for w in weights]
        h_model = Model()
        h_model.model.set_weights(h_weights)
        path = f"tests/tmp/healthy_{i}.keras"
        h_model.model.save(path)
        h_paths.append(path)
        
    # 3. Create 1 poisoned client (massive variance/outlier)
    p_weights = [w + np.random.normal(0, 2.0, size=w.shape) for w in weights]
    p_model = Model()
    p_model.model.set_weights(p_weights)
    p_path = "tests/tmp/poisoned.keras"
    p_model.model.save(p_path)
    
    # 4. Construct client_data tuples: (model_path, samples, loss, client_id)
    client_data = [
        (h_paths[0], 1000, 0.1, "client_0"),
        (h_paths[1], 1000, 0.1, "client_1"),
        (h_paths[2], 1000, 0.1, "client_2"),
        (p_path, 1000, 0.1, "client_poisoned")
    ]
    
    # 5. Run filter
    filtered = filter_poisoned_clients(client_data, global_path, mode="weights", threshold_multiplier=3.0)
    
    # Assertions
    filtered_ids = [item[3] for item in filtered]
    print(f"Filtered IDs: {filtered_ids}")
    
    assert "client_poisoned" not in filtered_ids, "Poisoned client was NOT filtered out!"
    assert len(filtered) == 3, f"Expected 3 clients, got {len(filtered)}"
    for idx in range(3):
        assert f"client_{idx}" in filtered_ids, f"Healthy client {idx} was incorrectly filtered!"
        
    print("Weights poisoning test passed!")
    
    # Cleanup temp files
    for path in h_paths + [p_path, global_path]:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists("tests/tmp"):
        os.rmdir("tests/tmp")

def test_gradients_poisoning_defense():
    print("\nTesting gradients poisoning defense...")
    # 1. Create a dummy base model to get shape structure
    model = Model()
    weights = model.model.get_weights()
    
    # 2. Create 3 healthy client gradients (small norms)
    h_grads = []
    for _ in range(3):
        h_grads.append([np.random.normal(0, 0.05, size=w.shape) for w in weights])
        
    # 3. Create 1 poisoned client gradient (large norm)
    p_grad = [np.random.normal(0, 10.0, size=w.shape) for w in weights]
    
    # 4. Construct client_data tuples: (client_grads, samples, loss, numeric_id)
    client_data = [
        (h_grads[0], 1000, 0.1, 0),
        (h_grads[1], 1000, 0.1, 1),
        (h_grads[2], 1000, 0.1, 2),
        (p_grad, 1000, 0.1, 999) # Poisoned client
    ]
    
    # 5. Run filter (global_model_path can be empty since it is unused in gradients mode)
    filtered = filter_poisoned_clients(client_data, global_model_path="", mode="gradients", threshold_multiplier=3.0)
    
    # Assertions
    filtered_ids = [item[3] for item in filtered]
    print(f"Filtered IDs: {filtered_ids}")
    
    assert 999 not in filtered_ids, "Poisoned client gradients were NOT filtered out!"
    assert len(filtered) == 3, f"Expected 3 clients, got {len(filtered)}"
    for idx in range(3):
        assert idx in filtered_ids, f"Healthy client gradient {idx} was incorrectly filtered!"
        
    print("Gradients poisoning test passed!")

if __name__ == "__main__":
    test_weights_poisoning_defense()
    test_gradients_poisoning_defense()
