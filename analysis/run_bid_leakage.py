"""Estimate how much vehicle type can be inferred from scalar DACA bids."""
import argparse
import pathlib
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.agents import DACAAgent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vehicles", type=int, default=3000)
    ap.add_argument("--sequence-length", type=int, default=30)
    ap.add_argument("--output", default="results/bid_leakage.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    types = rng.choice(3, size=args.vehicles, p=[0.3, 0.4, 0.3])
    params = [(20.0, 50_000.0), (15.0, 40_000.0), (10.0, 30_000.0)]
    X = []
    for typ in types:
        speed, battery = params[int(typ)]
        agent = DACAAgent(0, swarm_size=100)
        agent._speed, agent._max_energy = speed, battery
        agent._energy_rate = {20.0: 50.0, 15.0: 35.0, 10.0: 20.0}[speed]
        bids = []
        for _ in range(args.sequence_length):
            obs = np.array([0.0, 0.0, rng.uniform(.15, 1), rng.uniform(0, .8),
                            rng.uniform(0, 1), rng.uniform(.1, 1), rng.uniform(.2, 1),
                            rng.uniform(0, 1)], dtype=float)
            bids.append(agent.compute_bid(obs, exploration_noise=0.0))
        bids = np.asarray(bids)
        X.append([bids.mean(), bids.std(), np.mean(bids == 0),
                  np.quantile(bids, .25), np.quantile(bids, .75)])
    X = np.asarray(X)
    tr, te = train_test_split(np.arange(len(types)), test_size=.3, random_state=args.seed,
                              stratify=types)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X[tr], types[tr])
    pred = clf.predict(X[te])
    result = {"seed": args.seed, "vehicles": args.vehicles,
              "sequence_length": args.sequence_length,
              "accuracy": float(accuracy_score(types[te], pred)),
              "balanced_accuracy": float(balanced_accuracy_score(types[te], pred)),
              "chance_accuracy": 1/3}
    import json
    path = pathlib.Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
