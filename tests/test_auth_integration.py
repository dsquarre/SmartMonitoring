import sys
import os
import hashlib

# Append server directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from fastapi.testclient import TestClient
from main import app

def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def test_challenge_response_auth():
    test_client = TestClient(app)
    
    # 1. Test initiating auth with invalid client ID
    init_fail_res = test_client.post("/initiate", json={"client_id": "unknown_client"})
    assert init_fail_res.status_code == 401, f"Expected 401 for invalid client ID, got {init_fail_res.status_code}"
    print("Test 1 Passed: Unauthorized client ID registration rejected.")
    
    # 2. Test initiating auth with valid client ID (client_0)
    init_res = test_client.post("/initiate", json={"client_id": "client_0"})
    assert init_res.status_code == 200, f"Expected 200, got {init_res.status_code}"
    challenge = init_res.json().get("challenge")
    assert challenge is not None, "Expected challenge token to be returned"
    print(f"Test 2 Passed: Initiated authentication. Received challenge: {challenge}")
    
    # 3. Test authenticate with incorrect response (wrong password)
    wrong_pwd_hash = sha256("wrongpassword")
    wrong_response = sha256(wrong_pwd_hash + challenge)
    
    auth_fail_res = test_client.post("/authenticate", json={
        "client_id": "client_0",
        "response": wrong_response,
        "specs": {"cpu_frequency": 2.0e9}
    })
    assert auth_fail_res.status_code == 401, f"Expected 401 for wrong password, got {auth_fail_res.status_code}"
    print("Test 3 Passed: Rejected incorrect authentication response.")
    
    # 4. Re-initiate since the previous challenge is still active or needs refreshing
    init_res = test_client.post("/initiate", json={"client_id": "client_0"})
    challenge = init_res.json().get("challenge")
    
    # Test authenticate with correct response (password = P7h1!quiBO0no96)
    correct_pwd_hash = sha256("P7h1!quiBO0no96")
    correct_response = sha256(correct_pwd_hash + challenge)
    
    auth_success_res = test_client.post("/authenticate", json={
        "client_id": "client_0",
        "response": correct_response,
        "specs": {"cpu_frequency": 2.0e9}
    })
    assert auth_success_res.status_code == 200, f"Expected 200, got {auth_success_res.status_code}"
    assert auth_success_res.json().get("status") == "authenticated", "Expected authenticated status"
    print("Test 4 Passed: Authenticated successfully with correct password response.")
    
    # 5. Test replay attack (resubmitting the same response again)
    auth_replay_res = test_client.post("/authenticate", json={
        "client_id": "client_0",
        "response": correct_response,
        "specs": {"cpu_frequency": 2.0e9}
    })
    assert auth_replay_res.status_code == 400, f"Expected 400 for replay attack, got {auth_replay_res.status_code}"
    print("Test 5 Passed: Replay attack blocked successfully.")
    
    print("\nALL AUTHENTICATION INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    test_challenge_response_auth()
