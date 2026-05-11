# Federated Learning System for Healthcare 
Sponsored by AWS Amazon
- Compeletely abstracted code pipeline for Federated Learning

.
├── client
│   ├── dataset.npz
│   ├── main.py
│   ├── model.py
│   ├── psswd.txt
│   ├── requirements.txt
│   └── url.txt
│   
├── fl.yaml
├── LICENSE
├── README.md
└── server
    ├── main.py
    ├── model.py
    ├── ps.dat
    └── requirements.txt


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

## FL.YAML

- A template file for CloudFormation stack created in AWS
- 1 Server and 3 clients
- Almost everything is automated, just scp the dataset into each client
- ssh inside each client to start Federated Learning System
- Make sure the "ssh key" is replaced by actual ssh key obtained from terminal after doing 
```bash
ssh-keygen -t rsa -b 2048
```
- and replacing it with your key to be able to ssh into the ec2 instance.
- The shell commands in UsesData take roughly 5 minutes to execute.
- After 5 min, ssh into server and replace the ps.dat with actual hashed password you want to use, and ssh into clients to start the federated learning system.


## TODOs
- Error and edge cases handling
- Testing on AWS server and online clients to check convergence 

## How to Use

### Client
- Model definition is as defined in model.py and can be changed keeping the essence of funtions same.
- Upload the dataset and pass args to main.py; In our example we have preprocessed numpy arrays stored in dataset.npz
- Set the configs like server url and password and epochs per round and run main.py

```bash
#copy the dataset.npz here
#change password and url as required
pip install -r requirements.txt
python3 main.py -d dataset.npz
```

### Server
- Model definition must be safe as client.
- set the configs like rounds_left (no of rounds)
- run main.py through fastapi
- logs will be saved in server_log.txt
- store hashed password in ps.dat

```bash
pip install -r requirements.txt
fastapi run main.py
```
or 
```bash
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > server_log.txt 2>&1 &
#to kill
#sudo fuser -k 8000/tcp
```

