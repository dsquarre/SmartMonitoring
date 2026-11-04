# SmartMonitoring: distributed system

The deployable system, with the same architecture as before:
- FastAPI coordinator (`server/main.py`): authentication, WebSockets and the round loop.
- Redis: state and pub/sub.
- Celery worker (`server/celery_app.py`): runs `aggregate_models_task`.
- S3: models move through presigned URLs, or through the mock bucket.
- WebSocket clients (`client/`), the React dashboard and AWS CloudFormation.

Setup and running: [setup.md](setup.md) and [run.md](run.md). TLS: [deployment.md](deployment.md).

## Plugins

Every method is a plugin in `server/plugins/`, chosen by name. The system around the plugins is the same for every
method, and the simple simulator (`../simple`) imports the same package, so a method runs the same code in both.

| Slot | Setting | Plugins |
|---|---|---|
| Client selector | `FL_SELECTOR` | `oort` (default; the TMLR supplement's port), `random` |
| Server rule: one way to combine the client updates | used by the controllers; `FL_RULES` overrides a controller's set | `fedavg`, `fedyogi`, `median` (coordinate-wise), `krum` |
| Server controller: which rule each round, and whether to train on V | `FL_AGGREGATOR` | `select_train` (default, ours), `select`, `train`, `train_select`, `fedstrategist`, `sara`, or a rule name for that rule every round |
| Client behaviour | `FL_ATTACK`, `FL_ATTACK_FLIP` on a client | `none`, `standard` (5x the honest update); `stealth` is simulator-only, since it needs every client's update |

The controllers:
- `select`: the supplement's greedy-val. It builds every rule's candidate from the same updates and keeps the one with
  the lowest cross-entropy on the server's validation set V. Ties go to the first rule.
- `train`: the supplement's server-ft. One rule (`{"base": ...}`, default FedAvg), then plain SGD on V at the
  clients' learning rate and batch size.
- `select_train`: ours. Select as above, then train the kept model on V.
- `train_select`: the hybrid the supplement leaves open (§5.6). Every rule with and without training on V competes on
  V; set `FL_SELECT_FRAC` so a trained candidate is not scored on its own training data.
- `fedstrategist`: Haque et al. 2025, ported from the reproduction in `project x`. A LinUCB (`paper` preset) or
  CountUCB (`authors` preset) bandit picks FedAvg, median or Krum from the update diagnostics and learns from the
  change in validation accuracy minus the rule's cost. Arguments go in `FL_PLUGIN_ARGS`, e.g.
  `{"preset": "authors", "lambda_cost": 0.5}` or `{"reward_metric": "loss"}`. The accuracy reward is nearly flat on
  the imbalanced ECG data.
- `sara`: SARA (Hu et al., SEAMS 2025), as the TMLR paper describes it. UCB1 over the rules (default FedAvg, median,
  Krum) updates only the applied rule's estimate, with reward = accuracy − loss of the new global model on V.
  `{"c": 1.0}` sets the exploration constant. There is no public code, so this is a reconstruction.

Every controller reports the rule it kept and its server cost per round (`server_evals`: models scored on V;
`server_finetunes`: training passes over V). This goes to the dashboard history alongside the metrics. Controller
state, such as the FedYogi moments or the bandit, is stored next to the global model
(`models/global/server_state_<t>.pkl`), because each Celery task creates a fresh aggregator.

To add a method, write a class in `server/plugins/rules.py` or `server/plugins/controllers.py` with the registry
decorator (`@RULES.register("name")` or `@CONTROLLERS.register("name")`) and add a test to `tests/test_plugins.py`.
It is then available as `FL_AGGREGATOR=name` here and `--strategy name` in the simulator.

## What changed from the RL version

The infrastructure is unchanged: the processes, the Redis keys and pub/sub, the Celery task, S3 and presigned URLs,
the WebSocket protocol and the `.keras` model files. The learning logic follows the TMLR supplement
(`tmlr_supplement.zip`, `hrl_fl/`).

| Part | Before | Now |
|---|---|---|
| Client selection (`server/selector.py`) | LinUCB / WLS-TS / DQN agents, hierarchical meta-selector | the selector plugins. Oort registers clients with reward = training-set size and is updated with their statistical utility and measured round trip |
| `server/rl_env.py` | RL environment and reward normalisers | removed |
| Aggregation (`server/aggregator.py`) | FedAvg, qFedAvg, FedFV, FedAdam, FedProx, Krum, SCAFFOLD, MAD poisoning filter | an adapter that runs the controller plugins on the round's model files |
| Client training (`client/model.py`) | Keras `fit` with Adam (state carried across rounds), per-client class weights | plain SGD (no momentum, so no optimizer state between rounds), unweighted BCE, settings sent by the server with each `train` command. Oort's utility \|B\|·sqrt(mean per-sample loss²) comes from the losses seen during training |
| Client metrics | RL "soft" confusion counts; `accuracy` reported F2 | removed; `accuracy` is accuracy |

The ECG model has BatchNorm, which the supplement's CNN does not. FedYogi and the attacks act on the trainable weights;
the BatchNorm statistics take the FedAvg values.

V is the coordinating site's own labelled data: `SERVER_VAL_PATH`, an `.npz` file with `X` and `y`. For local runs,
`scripts/make_server_val.py` pools `non_iid` partitions 90–99, which no client uses.

## Testing with Docker

`docker-compose.yml` runs the whole stack using the unchanged `server/` and `client/` Dockerfiles: Redis, the
server, the Celery worker and four clients. LocalStack stands in for S3, so the real boto3 and presigned-URL code path
runs; set `S3_ENDPOINT_URL` to point `s3_helper.py` at it.

```bash
python scripts/make_server_val.py --src ../datasets/non_iid --clients 90-99 --out server/server_val.npz
docker compose up --build              # 3 rounds of select_train + Oort with 4 clients; dashboard on :8000
docker compose --profile tests run --rm tests   # pytest suite against the live Redis / S3 / Celery
docker compose down -v
```

To run another method, override the server and Celery environment in a second compose file. For example, set
`FL_AGGREGATOR: fedstrategist` and `FL_NUM_MALICIOUS: "1"` on `server` and `celery`, and `FL_ATTACK: standard` on
`client_3`, then run `docker compose -f docker-compose.yml -f override.yml up`.

`tests/test_plugins.py` needs only numpy and runs anywhere: `python -m unittest tests.test_plugins`.
