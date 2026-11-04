"""Run the simulation for one or more server strategies and seeds (sequentially; finished runs are skipped, interrupted
runs resume from their checkpoint). Strategies are the plugins of distributed/server/plugins:

    controllers  select_train (ours) | select | train | train_select | fedstrategist
    rules        fedavg | fedyogi | median | krum   (a rule name alone = that rule, every round)

    python run.py --strategy select_train --seeds 42                                   # our method, one seed
    python run.py --strategy fedavg,fedyogi,select,train,select_train --seeds 42-46 --out results/main
    python run.py --strategy train_select --select-frac 0.2 --seeds 42-46             # the hybrid of Sec. 5.6
    python run.py --strategy fedavg,median,krum,fedstrategist,select --rules-for-select fedavg,median,krum \
                  --num-malicious 20 --attack standard --seeds 42-46 --out results/poison
    python run.py --strategy fedstrategist --controller-args '{"preset": "authors"}' --name fs-authors
    python run.py --rounds 3 --num-clients 10 --clients-per-round 3                     # quick smoke run
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from smartmon.simulation import SimConfig, run
from plugins import CONTROLLERS, RULES, SELECTORS
from plugins.attacks import ATTACKS
from plugins.oort import OortConfig


def parse_seeds(text: str) -> list:
    seeds = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        seeds += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return seeds


def main():
    d = SimConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="select_train",
                   help=f"comma list; controllers: {', '.join(sorted(CONTROLLERS))}; rules: {', '.join(sorted(RULES))}")
    p.add_argument("--rules", default="", help="candidate rules for every controller (comma list); default: its own")
    p.add_argument("--rules-for-select", default="",
                   help="candidate rules for select / select_train / train_select only (e.g. fedavg,median,krum)")
    p.add_argument("--controller-args", default="{}", help='JSON, e.g. {"preset": "authors", "lambda_cost": 0.5}')
    p.add_argument("--rule-args", default="{}", help='JSON, e.g. {"krum": {"num_malicious": 2}}')
    p.add_argument("--name", default="", help="run label (one strategy only); default: the strategy")
    p.add_argument("--selector", default=d.selector, choices=sorted(SELECTORS))
    p.add_argument("--seeds", default="42", help="e.g. 42 | 42-46 | 42,50")
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--out", default="results/run")
    p.add_argument("--num-clients", type=int, default=d.num_clients, help="0 = every client file")
    p.add_argument("--clients-per-round", "-k", type=int, default=d.clients_per_round)
    p.add_argument("--rounds", "-r", type=int, default=d.rounds)
    p.add_argument("--local-epochs", "-e", type=int, default=d.local_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr, help="client SGD learning rate")
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--val-size", type=int, default=d.val_size, help="subsample V (0 = all)")
    p.add_argument("--val-pos-share", type=float, default=d.val_pos_share,
                   help="with --val-size: draw V with this AFib share (label-skewed server set)")
    p.add_argument("--normalize", default=d.normalize, choices=["none", "record"], help="signal preprocessing")
    p.add_argument("--server-lr", type=float, default=d.server_lr, help="FedYogi eta")
    p.add_argument("--ft-epochs", type=int, default=d.ft_epochs, help="server training epochs on V")
    p.add_argument("--ft-lr-scale", type=float, default=d.ft_lr_scale, help="server lr = client lr x this")
    p.add_argument("--select-frac", type=float, default=d.select_frac,
                   help="share of V reserved for scoring candidates (rest: training only); 0 = all of V for both")
    p.add_argument("--num-malicious", type=int, default=d.num_malicious, help="malicious clients (fixed per seed)")
    p.add_argument("--attack", default=d.attack, choices=ATTACKS)
    p.add_argument("--attack-scale", type=float, default=d.attack_scale)
    p.add_argument("--flip", action="store_true", help="reverse the malicious update direction")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--log-every", type=int, default=0, help="print every N-th round (default: rounds 1-3, every 10th)")
    a = p.parse_args()

    strategies = [s.strip() for s in a.strategy.split(",") if s.strip()]
    if a.name and len(strategies) > 1:
        p.error("--name needs a single --strategy")
    for strategy in strategies:
        rules = a.rules
        if a.rules_for_select and strategy in ("select", "select_train", "train_select"):
            rules = a.rules_for_select
        for seed in parse_seeds(a.seeds):
            cfg = SimConfig(data_dir=a.data_dir, num_clients=a.num_clients, clients_per_round=a.clients_per_round,
                            rounds=a.rounds, local_epochs=a.local_epochs, batch_size=a.batch_size, lr=a.lr,
                            val_fraction=a.val_fraction, val_size=a.val_size, val_pos_share=a.val_pos_share,
                            normalize=a.normalize, seed=seed, selector=a.selector,
                            strategy=strategy, rules=tuple(r for r in rules.split(",") if r),
                            rule_args=json.loads(a.rule_args), controller_args=json.loads(a.controller_args),
                            server_lr=a.server_lr, ft_epochs=a.ft_epochs, ft_lr_scale=a.ft_lr_scale,
                            select_frac=a.select_frac, num_malicious=a.num_malicious, attack=a.attack,
                            attack_scale=a.attack_scale, flip=a.flip, name=a.name, oort=OortConfig())
            print(run(cfg, a.out, verbose=not a.quiet, log_every=a.log_every), flush=True)


if __name__ == "__main__":
    main()
