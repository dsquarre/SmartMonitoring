# SmartMonitoring Deployment & SSL Configuration Guide

This guide covers deploying the **SmartMonitoring** Federated Learning server and clients with **HTTPS** and **WSS (WebSocket Secure)**.

---

## Important SSL/TLS Rule: IP Addresses vs Domain Names

> [!IMPORTANT]
> **Certbot (Let's Encrypt)** requires a registered domain name (e.g., `api.yourdomain.com`). Let's Encrypt **does not** issue certificates for raw public or private IP addresses (e.g., `1.2.3.4`).
>
> - **If deploying with a raw IP address**: Use **Self-Signed SSL Certificates** generated via `openssl`.
> - **If deploying with a Domain Name**: Use **Certbot (Let's Encrypt)**.
> - **If deploying on AWS (Recommended)**: Use an **AWS Application Load Balancer (ALB)** with AWS Certificate Manager (ACM).

---

## 1. IP-Based Deployment (Self-Signed SSL)

Use this method when deploying directly to an AWS EC2 instance or local server using a raw IP address.

### Step 1: Generate Self-Signed Certificates
Run the following command on your server to create a 2048-bit RSA key and self-signed certificate:

```bash
mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/server.key \
  -out certs/server.crt \
  -subj "/CN=<YOUR_SERVER_IP>"
```

### Step 2: Start the FastAPI Server with SSL
Pass the certificate and key flags to Uvicorn:

```bash
# In server/ directory
S3_MOCK=true uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-keyfile ../certs/server.key \
  --ssl-certfile ../certs/server.crt
```

### Step 3: Extract Certificate SHA-256 Fingerprint
On the server, run the following command to get the certificate fingerprint:

```bash
openssl x509 -noout -fingerprint -sha256 -in certs/server.crt
# Output: sha256 Fingerprint=12:0A:9E:87:19:A3:23:AF:49:4D:54:40:...
```

### Step 4: Launch Clients with Certificate Pinning (Recommended / MitM Safe)
Pass the fingerprint via `-f` / `--fingerprint` (or set `CERT_FINGERPRINT` in `.env`). This provides **100% security against Man-in-the-Middle (MitM) attacks** without needing a domain name or paid CA:

```bash
# In client/ directory
python main.py \
  -s https://<YOUR_SERVER_IP>:8000 \
  -c client_0 \
  -p P7h1!quiBO0no96 \
  -d /path/to/dataset.npz \
  -f 12:0A:9E:87:19:A3:23:AF:49:4D:54:40:67:87:80:CC:80:02:78:45:74:75:86:38:6D:5D:C3:94:67:E1:F2:7A
```
*The client will verify the certificate fingerprint during SSL handshake. If an attacker attempts to intercept the connection with a fake certificate, the client will immediately drop the connection with `SSL Fingerprint Mismatch`.*

---

## 2. Domain-Based Deployment (Certbot & Let's Encrypt)

Use this method if you have mapped a domain name (e.g., `fl.yourdomain.com` or dynamic DNS like `nip.io`) to your server IP.

### Step 1: Install Certbot and Obtain Certificate
```bash
sudo apt-get update && sudo apt-get install -y certbot
sudo certbot certonly --standalone -d fl.yourdomain.com
```
*Certbot will save certificates to `/etc/letsencrypt/live/fl.yourdomain.com/`.*

### Step 2: Start FastAPI Server with Let's Encrypt
```bash
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-keyfile /etc/letsencrypt/live/fl.yourdomain.com/privkey.pem \
  --ssl-certfile /etc/letsencrypt/live/fl.yourdomain.com/fullchain.pem
```

### Step 3: Launch Clients with Verified SSL
```bash
python main.py \
  -s https://fl.yourdomain.com:8000 \
  -c client_0 \
  -p P7h1!quiBO0no96 \
  -d /path/to/dataset.npz
```

---

## 3. AWS Production Deployment (ALB + ACM - Recommended)

For enterprise AWS deployments, offload SSL termination to AWS infrastructure so your Docker containers run cleanly without certificate files.

```text
[Client Device] --- (HTTPS / WSS on Port 443) ---> [AWS ALB] --- (HTTP / WS on Port 8000) ---> [EC2 / Docker Container]
```

### Setup Steps:
1. **Request Certificate**: In **AWS Certificate Manager (ACM)**, request a free public certificate for your domain (`*.yourdomain.com`).
2. **Create ALB**: Create an AWS Application Load Balancer with an HTTPS listener on Port 443 attached to your ACM certificate.
3. **Configure Target Group**: Point the ALB listener to your EC2 instance on Port 8000.
4. **Deploy Containers**: Run your server container standard HTTP (no `--ssl-*` flags needed).

---

## 4. Docker Compose Deployment with SSL

To deploy server and client containers using Docker Compose with self-signed SSL:

### `docker-compose.yml`
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  server:
    build:
      context: ./server
    ports:
      - "8000:8000"
    volumes:
      - ./certs:/app/certs
    environment:
      - FL_N=10
      - FL_K=3
      - S3_MOCK=true
      - SERVER_HOST=https://<YOUR_SERVER_IP>:8000
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile /app/certs/server.key --ssl-certfile /app/certs/server.crt

  client:
    build:
      context: ./client
    environment:
      - SERVER_IP=https://server:8000
      - CLIENT_ID=client_0
      - PASSWORD=P7h1!quiBO0no96
      - NO_VERIFY=true
    command: python main.py -d /data/dataset.npz
```
