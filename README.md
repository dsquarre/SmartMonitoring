# Federated Learning System for Healthcare 
Sponsored by AWS Amazon
- Compeletely abstracted code pipeline for Federated Learning

Repository is divided into two dirs
- client/
- server/


## Client Side 

- Clients control the flow of execution of code. Each client that wants to participate in the system first authenticates itself to get a client_id.
- Using this Id it can upload its local training weights and evaluation report. 
- Downloading the global model or getting version info does not require client id and hence anyone can get them.
- Authentication is done in order to prevent model poisoning .
- Client pulls the global model from server, trains on local data for a few epochs and uploads the local model to server.
- Waits for new version of global model to be available or rounds to finish.
- After rounds, it evaluates on local data and uploads it on the server.
- Fetches the global metrics and logs it.


## Server Side

- Server is deployed on AWS cloud and runs on a uvicorn server which constantly waits for users to authenticate and upload their weights
- A round consists of N>=3 clients uploading their weights (it starts a background check for aggregate function), after which FedAvg algorithm w_i_t+1 = sigma(t=1,n,(Ni*w_t/N)) is used to calculate global weights and saved in global_model_path and current_round is incremented.
- Clients regularily ping the server to check if the server's current_round is different from their local current round, if yes then they download the latest global_model file and start the next round.
- This is repeated until rounds_left becomes 0 after which clients start their local evaluation of the model and upload to the server(starting a background check for aggregate function). When N>=3 authenticated clients have uploaded evaluation metrics, FedAvg of client metrics is done and global_metrics is available to download by setting done=True.
- Clients regularily ping the server/done and if its true, they download the global metrics.


## TODOs
- Authentication 
- Error and edge cases handling
- Testing on AWS server and online clients to check convergence 

## How to Use

### Client
- Model definition is as defined in model.py and can be changed keeping the essence of funtions same.
- Upload the dataset and pass args to main.py; In our example we have preprocessed numpy arrays stored in data/dataset.npz
- Set the configs like server url and epochs and run main.py

```bash
mkdir data
#copy the dataset.npz in data dir
pip install -r requirements.txt
python3 main.py -d data/dataset.npz
```

### Server
- Model definition must be safe as client.
- set the configs like rounds_left (no of rounds)
- run main.py through fastapi
- logs will be saved in server_log.txt

```bash
pip install -r requirements.txt
fastapi run main.py > server_log.txt 2>&1 &
```
