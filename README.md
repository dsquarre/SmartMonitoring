# SmartMonitoring: Centralized Federated Learning for Healthcare

An end-to-end, scratch-implemented distributed Federated Learning (FL) framework optimized for privacy-preserving atrial-fibrillation detection on edge ECG data. Built to empirically study client selection behaviors under heavy, non-IID data skew. 
*Sponsored by AWS Amazon; manuscript in preparation for TMLR.*

## Key Highlights & Architecture
* **Custom Topology:** Orchestrated a centralized coordination system running **100 networked Docker clients deployed across 10 AWS EC2 instances**.
* **Zero-Library Math:** Hand-implemented mathematical optimization frameworks directly from literature: `FedAvg`, `FedFV`, `FedAdam`, `q-FedAvg`, and `FedProx`.
* **Asynchronous Networking:** Architected client-server coordination over persistent **FastAPI WebSockets** using SHA-256 certificate pinning and token authentication.
* **Hardware Efficiency:** Implemented a sequential client execution runner on edge nodes to seamlessly evaluate multiple clients while keeping peak memory utilization below target constraints.

## System Components
### Coordinator Server (`/server`)
* Enforces password-based SHA-256 handshake protocols to establish valid websocket sessions.
* Orchestrates multi-round training loops, applying weights aggregation over custom network protocols once the $N$-client threshold is met.
* Executes advanced contextual client-selection policies: `Random`, `LinUCB`, `DQN`, and `Hierarchical RL`.

### Edge Client (`/client`)
* Decoupled runtime structure that continuously listens for `train`, `wait`, or `stop` triggers.
* Downloads global weights, trains local backends (TensorFlow/PyTorch), pipes metrics up, and generates localized evaluation convergence trajectories.
