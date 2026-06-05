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


## TODOs
- Error and edge cases handling
- check convergence

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
