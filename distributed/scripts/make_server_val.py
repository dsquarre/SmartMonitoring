"""Builds the server's validation set V (an .npz with X, y) for the select / train / select_train strategies.

V is the coordinating site's own labelled ECG data. For local runs it is made from whole client partitions
that no participating client uses (by default non_iid partitions 90-99, train and test splits), so V is
disjoint from every client's data. Launch clients only from partitions outside that range.

    python scripts/make_server_val.py --src ../datasets/non_iid --clients 90-99 --out server/server_val.npz
"""
import argparse
import os

import numpy as np


def parse_range(text: str):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="../datasets/non_iid", help="folder with client_<i>.npz partitions")
    parser.add_argument("--clients", default="90-99", help="partitions to pool, e.g. 90-99 or 3,7,9")
    parser.add_argument("--out", default="server/server_val.npz")
    args = parser.parse_args()

    xs, ys = [], []
    for i in parse_range(args.clients):
        path = os.path.join(args.src, f"client_{i}.npz")
        with np.load(path) as d:
            for split in ("train", "test"):
                xs.append(np.asarray(d[f"X_{split}"], dtype=np.float32))
                ys.append(np.asarray(d[f"y_{split}"], dtype=np.float32).reshape(-1))
    X, y = np.concatenate(xs), np.concatenate(ys)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, X=X, y=y)
    print(f"Server validation set: {len(y)} records ({int(y.sum())} AFib) from partitions {args.clients} -> {args.out}")


if __name__ == "__main__":
    main()
