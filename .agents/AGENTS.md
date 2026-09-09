# Project Rules for SmartMonitoring

## Client Cost Profile Rules
- When actual training metrics (latency/energy) are not available, default them to `0.0` instead of using theoretical cycle/frequency formulas to classify client cost profiles in `rl_env.py` and `selector.py`.

## State Feature Default Rules
- Unobserved/uninitialized loss, accuracy, latency, energy, and initial staleness state features in $X_t$ (Client Selector) and $G_t$ (Meta-Aggregator) default to `np.inf` / `float('inf')`.

## CloudFormation Syntax Rules
- In CloudFormation YAML templates for EC2 `UserData`, use `UserData: !Base64 |` directly without inserting `Fn::Base64: !Sub |` or placing `!Sub` after it.
