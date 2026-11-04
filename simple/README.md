# SmartMonitoring: simple (single-process) version

Federated learning for AFib detection on 12-lead ECG (PTB-XL), simulated in one process: no Redis, Celery, S3 or
network. The methods (client selectors, server rules and controllers, attacks) are the plugins of the distributed
system, `../distributed/server/plugins`, so a result here is a result of the code the deployed system runs.

```
simulate_fl.py          the main file: data, model, training, simulation loop, plots, CLI
compare_experiments.py  paired comparisons across seeds (t-test, Wilcoxon, Holm) and comparison plots
EXPERIMENTS.md          CLI reference and every hypothesis with its commands, run one at a time
experiments/ecg_protocol.md   pre-declared protocol: calibration, hypotheses, decision rules, amendments
tests/                  python -m unittest discover -s tests
```

Quick start (from `simple/`, with the conda env active so TensorFlow uses the GPU):

```bash
conda activate py310
python simulate_fl.py -a select --seed 42 -r 5 -o results/test          # a 5-round check
python simulate_fl.py -a select -o results/h1_select                     # base: 120 rounds, seeds 42-71
python compare_experiments.py --dirs results/h1_select results/h1_fedavg results/h1_fedyogi --labels Selector FedAvg FedYogi --best-of FedAvg,FedYogi -o results/hypothesis1
```

Base parameters (defaults): 120 rounds, 5 local epochs, seeds 42-71, 20 of 100 clients per round with Oort, the
label-skewed split `../datasets/non_iid`, plain-SGD clients at lr 0.1, FedYogi step 0.01, per-record normalisation.
The server's validation set V and the reported test set T are the pooled client test splits, halved per seed; T is
never used for a decision. Calibration and every decision rule: `experiments/ecg_protocol.md`.
