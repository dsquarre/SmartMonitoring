# SmartMonitoring: Porting Guide & TODO for Main Application

This document tracks all features, metrics, and protocols developed and validated in `simulate_fl.py` that need to be integrated into the distributed production application (`server/main.py`, `client/main.py`, `server/rl_env.py`, `server/aggregator.py`).

---

## Completed in Core Models (`client/model.py` & `server/model.py`)
- [x] **Soft Confusion Metrics**: `ClientModel.evaluate()` computes probabilistic continuous metrics (`soft_tp`, `soft_fp`, `soft_tn`, `soft_fn`, `soft_precision`, `soft_recall`, `soft_f2`) alongside hard-threshold (0.3) clinical reporting metrics.
- [x] **Round 0 Metadata Handshake Support**: `ClientModel.get_class_counts()` added to extract local train dataset total and positive sample counts without privacy leakage.
- [x] **Dynamic Output Bias & Class Weight Configuration**: `ClientModel` and `ServerModel` accept dynamic `initial_bias` (for output sigmoid bias initialization) and `class_weights` (for training).
- [x] **Gradient & Weight Update L2 Norm Tracking**: Computed global, mean client, and max client update $L_2$ norms (`global_grad_norm`, `mean_client_grad_norm`, `max_client_grad_norm`) with NaN/Inf numerical safety guards and saved `grad_norm_vs_round.png` plot.

---

## TODO: Server Application (`server/main.py`, `server/aggregator.py`, `server/rl_env.py`)

### 1. Round 0 Handshake Protocol (`server/main.py`)
- [ ] **Registration & Metadata Collection**:
  - During client registration or pre-training Round 0, collect `total_samples` ($N_i$) and `pos_samples` ($N_{i, \text{pos}}$) from each client.
- [ ] **Global Prior & Class Weight Calculation**:
  - Compute global positive class ratio: $p_{\text{global}} = \frac{\sum N_{i, \text{pos}}}{\sum N_i}$.
  - Calculate global output bias: $b_{\text{global}} = \ln \left( \frac{p_{\text{global}}}{1 - p_{\text{global}}} \right)$.
  - Calculate global class weights: $w_{\text{global, pos}} = \frac{N_{\text{global}}}{2 N_{\text{global, pos}}}, w_{\text{global, neg}} = \frac{N_{\text{global}}}{2 N_{\text{global, neg}}}$.
- [ ] **Global Model Build**:
  - Instantiate `ServerModel(initial_bias=global_bias)` so the initial broadcast weights $W_0$ have $b_{\text{global}}$ baked in.

### 2. Soft Confusion Micro-Aggregation & RL Reward Signal (`server/main.py` & `server/rl_env.py`)
- [ ] **Server Micro-Aggregation**:
  - Update server metric collection endpoint to extract `soft_tp`, `soft_fp`, `soft_tn`, `soft_fn` from client response payloads.
  - Micro-aggregate global soft counts:
    $$\text{Global Soft-}F_2 = \frac{5 \cdot \text{Soft-TP}}{5 \cdot \text{Soft-TP} + 4 \cdot \text{Soft-FN} + \text{Soft-FP}}$$
- [ ] **Meta-Aggregator Reward Alignment (`server/rl_env.py`)**:
  - Use continuous $\Delta \text{Soft-}F_2 = \text{Soft-}F_{2, t} - \text{Soft-}F_{2, t-1}$ for the Meta-Controller accuracy reward component instead of hard-threshold accuracy delta.
  - Ensure the reward pipeline uses Z-score normalization $\to$ weighted combination $\to$ tanh squashing to satisfy bounded D-LinUCB regret conditions.

### 3. Client Cost Profile Default Rules (`server/rl_env.py` & `server/selector.py`)
- [ ] **Default Metrics Alignment**:
  - When actual wall-clock training metrics (latency/energy) are unobserved or uninitialized, default them to `0.0` (per `AGENTS.md` project rules) rather than using theoretical hardware formulas.
  - Default unobserved accuracy/loss state features in $X_t$ and $G_t$ to `np.inf` / `float('inf')`.

---

## TODO: Client Application (`client/main.py`)

### 1. Round 0 Handshake Endpoint (`client/main.py`)
- [ ] **Send Metadata Counts**:
  - Update client startup/registration payload to include `get_class_counts()` output (`total_samples`, `pos_samples`).

### 2. Local Training & Evaluation Updates (`client/main.py`)
- [ ] **Global Class Weights Local Training**:
  - Receive `global_class_weights` from server Round 0 response and initialize `ClientModel(..., class_weights=global_class_weights)`.
- [ ] **Soft Confusion Payload Transmission**:
  - Send `soft_tp`, `soft_fp`, `soft_tn`, `soft_fn` in the evaluation response JSON sent back to the server.

---

## Verification & Testing
- [ ] Run full end-to-end integration test with 1 server instance and 2 client containers.
- [ ] Verify Round 0 log prints computed $b_{\text{global}}$ and $w_{\text{global, pos}}$.
- [ ] Verify server receives `soft_f2` metrics and logs continuous meta-rewards.
