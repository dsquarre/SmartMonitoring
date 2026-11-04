# Project Rules for SmartMonitoring

## Plugins
- Every method (client selector, server rule, server controller, client attack) is a plugin in `distributed/server/plugins/`, numpy-only, registered by name. It is the only implementation: the distributed system (`server/selector.py`, `server/aggregator.py`) and the simulator (`simple/smartmon`, which loads the package as `plugins`) both use it. Do not reimplement a method in either host.
- Model-specific work (scoring on V, training on V) reaches a controller only through the `RoundContext` callbacks `evaluate` and `finetune`.
- `distributed/client/attacks.py` must stay a byte-identical copy of `distributed/server/plugins/attacks.py` (checked by `distributed/tests/test_plugins.py`).
- Keep the distributed infrastructure unchanged (Redis, Celery task, S3, WebSocket protocol, `.keras` files); new methods are new plugins.
- `plugins/oort.py` is a faithful port of the reference Oort selector; keep its documented quirks unless a change is deliberate and documented in its header.

## Client Cost Profile Rules
- When actual training metrics (latency/energy) are not available, report them as `0.0` rather than inventing values; the simulation's cost model (`simple/smartmon/profiles.py`) is the only place theoretical device formulas are used.

## CloudFormation Syntax Rules
- In CloudFormation YAML templates for EC2 `UserData`, use `UserData: !Base64 |` directly without inserting `Fn::Base64: !Sub |` or placing `!Sub` after it.
