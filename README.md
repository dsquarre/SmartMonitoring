# Federated Learning System for Healthcare 
Sponsored by AWS Amazon
- Compeletely abstracted code pipeline for Federated Learning
```
.
├── client
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
```

## Client Side 

- Clients control the flow of execution of code. Each client that wants to participate in the system first authenticates itself with a password to get a client_id
- Using this Id it can upload its local training weights and evaluation report
- Downloading the global model or getting version info does not require client id and hence no id is required for those methods
- Authentication is done in order to prevent model poisoning attacks
- To simulate multiple clients on low RAM, we used sequential client loading and updating
- Each clients data is stored as C0.npz, C1.npz etc so the argument should only be the directory and starting name of the data files
- Each client obj with its own model and id is stored in a list and each client trains and uploads its model one by one till all the uploads are done or no rounds are left
- Then after a new global version is available, all clients download the global model and reinitialize their model weights
- After all rounds are done, all clients one by one evaluate the model on their local data and upload local metrics on the server
- Get the global metrics and logs it after its done


## Server Side

- Server is deployed on AWS cloud and runs on a uvicorn server which constantly waits for users to authenticate and upload their weights
- A round consists of N>=10 clients uploading their weights (it starts a background check for aggregate function), after which FedAvg algorithm w_i_t+1 = sigma(t=1,n,(Ni*w_t/N)) is used to calculate global weights and saved in global_model_path and current_round is incremented
- To avoid stale update problem, clients upload their round and server checks if it matches the current round, if not the model file is dropped else it is queued
- We set the current version to higher to avoid clients from asynchronously uploading model files while aggregation is happening and after aggregation, client modle queue is cleared to start receiving fresh updates
- Clients regularily ping the server to check if the server's current_round is different from their local current round, if yes then they download the latest global_model file and start the next round
- This is repeated until rounds_left becomes 0 after which clients start their local evaluation of the model and upload to the server(starting a background check for aggregate function). When N>=10 authenticated clients have uploaded evaluation metrics, FedAvg of client metrics is done and global_metrics is available to download by setting done=True.
- Clients regularily ping the server/done and if its true, they download the global metrics.

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


## TODOs
- Error and edge cases handling
- Testing on AWS server and online clients to check convergence
- Writing tests for stale model and metrics update and if the server drops them as intended

## How to Use

### Client
- Model definition is as defined in model.py and can be changed keeping the essence of funtions same.
- Upload the dataset in A/ or B/ or C/. In our example we have preprocessed numpy arrays stored in A/A0.npz B/B0.npz etc
- Set the configs like server url and password and epochs per round and run main.py

```bash
#change password and url as required
mkdir A B C
#copy all the datasets for example C0.npz, C1.npz till no of clients
pip install -r requirements.txt
#python3 main.py -d C/C > client_log.txt
#or
chmod +x run.sh
./run.sh
```

### Server
- Model definition must be same as client.
- set the configs like rounds_left (i.e. no of rounds)
- store hashed password in ps.dat
- run main.py through fastapi


```bash
pip install -r requirements.txt
fastapi run main.py > server_log.txt
```
or 
```bash
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > server_log.txt 2>&1 &
#to kill
#sudo fuser -k 8000/tcp
```

