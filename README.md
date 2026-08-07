# Federated Learning System for Healthcare 
Sponsored by AWS Amazon

## Client Side 

- Clients ping server with a password to get client id.
- Each client opens a websocket at ws/ip:8000/ws/client_id and recieves commands from server to either "train" or "wait" or "stop". 
- Each client that recieves train downloads global model, start training, upload file to server other clients wait
- To preserve RAM client.py runs multiple sequential clients that train and evaluate one by one instead of parallely.
- Then each client gets pinged to start evaluation and evaluate model on their local test data and upload metrics to server
- Each client also makes a local & global metrics vs round plot to check convergence


## Server Side

- Generates a random id for each user on recieving password
- Allows only authenticated users to start websocket
- Once N clients have started socket, start fd round. 
- For now, all clients recieve "train" command
- Applies simple FedAvg algorithm to update global model and pings each client when done
- When all clients have uploaded evaluation metrics, again apply FedAvg on metrics to get global metrics and start next round till all rounds are done

## FL.YAML

- A template file for CloudFormation stack created in AWS
- 1 Server and 2 client instances (which run multiple sequential and parallel clients)
- Almost everything is automated, just scp the dataset into each client instance
- then ssh inside each instance to run main.py
- Make sure the "ssh key" is replaced by actual ssh key obtained from terminal after doing 
```bash
ssh-keygen -t rsa -b 2048
```
- and replacing it with your key to be able to ssh into the ec2 instance
- The shell commands in UserData take roughly 5 minutes to execute
- After 5 min, ssh into server and replace the ps.dat with actual hashed password you want to use, and ssh into clients to start the federated learning system


## How to Add and Configure a New Client

To add a new client (e.g. `client_11`) to the Federated Learning system:

### 1. Register Credentials on the Server
Generate the SHA256 hash of the client's desired password:
```bash
python -c "import hashlib; print(hashlib.sha256(b'YOUR_PASSWORD').hexdigest())"
```
Add the client ID and hash to [server/credentials.json](file:///home/danish/SmartMonitoring/server/credentials.json):
```json
{
  "client_0": "8b841ea...",
  "client_11": "your_generated_sha256_hash_here"
}
```

### 2. Start the Client Node
Start the client with the matching client ID and password:
```bash
python client/main.py -d /path/to/dataset.npz -s <server_ip_or_url> -p <password> -c client_11
```

---

## How to Use

### Client
* **Parameters**:
  - `-d, --dataset`: (Required) Path to the `.npz` local dataset.
  - `-s, --server-ip`: Server IP or URL (e.g. `127.0.0.1:8000` or `https://my-fl-server.com`).
  - `-p, --password`: Password configured for the client ID.
  - `-c, --client-id`: The unique client ID registered in `credentials.json` (Defaults to `client_0`).
  - `--no-verify` / `--insecure`: Bypass SSL certificate checks (useful for testing self-signed SSL/TLS setups).

```bash
# Example running locally
pip install -r client/requirements.txt
python client/main.py -d data/C0.npz -s localhost:8000 -p P7h1!quiBO0no96 -c client_0
```

### Server
* Ensure your hashed client credentials are set in [server/credentials.json](file:///home/danish/SmartMonitoring/server/credentials.json).
* Run the server using uvicorn:

```bash
pip install -r server/requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

