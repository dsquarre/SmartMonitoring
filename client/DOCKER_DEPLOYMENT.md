# Docker Deployment and Simulation Guide

This guide describes how to build and run the simplified Federated Learning client inside Docker containers. This setup allows running multiple independent clients on the same machine (like a local PC or an AWS EC2 instance) while simulating different hardware profiles.

## 1. Building the Docker Image
To build the Docker image, run the following command in the `client/` directory:
```bash
docker build -t fl-client .
```

*Note: The Dockerfile is optimized to exclude TensorFlow from the pip installation phase since it is already pre-installed in the base TensorFlow image. This significantly speeds up the build process.*

## 2. Running a Single Client
To run a containerized client, pass the server connection details, client identity, and mount your local dataset directory into the container.

```bash
docker run -d \
  -e SERVER_IP="<SERVER_IP>:<PORT>" \
  -e PASSWORD="<PASSWORD>" \
  -e CLIENT_ID="client_0" \
  -v /path/to/local/data:/data \
  fl-client -d /data/A0.npz
```

* **`-e SERVER_IP`**: Specifies the server address (e.g. `172.31.0.10:8000` or `10.0.0.5:8000`). Supports both HTTP/WS and HTTPS/WSS (e.g. `https://my-server.com`).
* **`-e PASSWORD`**: Specifies the password to authenticate with the server.
* **`-e CLIENT_ID`**: The unique client ID registered in `credentials.json` on the server.
* **`-v /path/to/local/data:/data`**: Mounts the folder containing your `.npz` dataset files.
* **`fl-client -d /data/A0.npz`**: Loads and trains on `A0.npz`. You can append `--no-verify` to bypass SSL checks if using self-signed certs: `fl-client -d /data/A0.npz --no-verify`.

---

## 3. Simulating Heterogeneous Clients on AWS EC2 (or Single Host)
You can spawn multiple independent client containers on a single host. To simulate heterogeneous clients (e.g., Raspberry Pi vs. high-end PC), restrict the CPU cores and memory limits of each container.

### Example A: Simulate a Weak Client (e.g., Raspberry Pi)
Limit the container to **1 CPU Core** and **1GB RAM**:
```bash
docker run -d \
  --cpus="1.0" \
  --memory="1g" \
  -e SERVER_IP="172.31.0.10:8000" \
  -e PASSWORD="P7h1!quiBO0no96" \
  -v /home/ubuntu/data:/data \
  fl-client -d /data/A1.npz
```

### Example B: Simulate a Medium Client (e.g., Standard Office PC)
Limit the container to **2 CPU Cores** and **2GB RAM**:
```bash
docker run -d \
  --cpus="2.0" \
  --memory="2g" \
  -e SERVER_IP="172.31.0.10:8000" \
  -e PASSWORD="P7h1!quiBO0no96" \
  -v /home/ubuntu/data:/data \
  fl-client -d /data/A2.npz
```

### Example C: Simulate a Strong Client (e.g., High-end Workstation)
Limit the container to **4 CPU Cores** and **4GB RAM**:
```bash
docker run -d \
  --cpus="4.0" \
  --memory="4g" \
  -e SERVER_IP="172.31.0.10:8000" \
  -e PASSWORD="P7h1!quiBO0no96" \
  -v /home/ubuntu/data:/data \
  fl-client -d /data/A3.npz
```

---

## 4. How CodeCarbon Works in this Setup
* **Unprivileged Mode (Default):** CodeCarbon automatically falls back to CPU lookup database matching and TDP estimation based on the container's active CPU load. This is highly portable and ensures each client reports its own separate energy footprint.
* **AWS EC2 Auto-Detection:** If run on an AWS EC2 instance, CodeCarbon automatically queries the AWS Instance Metadata service to detect the precise EC2 instance type (e.g., `t3.medium`) and dynamically estimates the exact power consumption of that instance without requiring any extra setup.
