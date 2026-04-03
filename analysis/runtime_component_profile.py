"""Profile runtime components with real simulator/agents and export runtime_breakdown.csv."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import AgentPool, DACAConfig
from src.simulator import SwarmSimulator

METHODS = ["daca", "qlearning", "auction_nolearning", "greedy"]
METHOD_MAP = {
    "daca": "DACA",
    "qlearning": "Q-learning",
    "auction_nolearning": "AL",
    "greedy": "Greedy",
}


def profile(method: str, n: int, tasks: int, seed: int, device: str) -> Dict:
    sim = SwarmSimulator(num_drones=n, task_arrival_rate=2.0, duration=max(200.0, tasks / 2.0 + 5.0), seed=seed)
    pool = AgentPool(n, agent_type=method, daca_config=DACAConfig(device=device))

    t_bid = 0.0
    t_auc = 0.0
    t_upd = 0.0
    cnt = 0

    for step in range(tasks):
        task = sim.next_task()
        if task is None:
            break
        obs = {d.drone_id: sim.get_observation(d, task) for d in sim.drones}

        eps = 0.02 * (0.9996 ** step) if method == "daca" else (0.10 * (0.999 ** step) if method == "qlearning" else 0.0)

        t0 = time.perf_counter()
        bids = pool.compute_bids(obs, exploration_noise=eps)
        t1 = time.perf_counter()
        result = sim.run_auction(task, bids)
        t2 = time.perf_counter()

        t_bid += (t1 - t0)
        t_auc += (t2 - t1)

        if method in ("daca", "qlearning"):
            truthful = result.get("truthful_bids", {})
            rewards = result.get("rewards", {})
            winner = result.get("winner_id", None)
            u0 = time.perf_counter()
            for drone_id, state in obs.items():
                reward = float(rewards.get(drone_id, 0.0))
                if winner is not None:
                    reward += 0.10 * float(task.priority / 100.0) if drone_id == winner else 0.02 * float(task.priority / 100.0)
                pool.update_agent(
                    drone_id,
                    state,
                    float(max(0.0, bids.get(drone_id, 0.0))),
                    reward,
                    state,
                    False,
                    truthful_bid=float(truthful.get(drone_id, 0.0)),
                    winner_id=winner,
                )
            u1 = time.perf_counter()
            t_upd += (u1 - u0)

        cnt += 1

    if cnt == 0:
        cnt = 1

    return {
        "method": METHOD_MAP[method],
        "swarm_size": int(n),
        "bid_compute_ms": float((t_bid / cnt) * 1000.0),
        "auction_resolve_ms": float((t_auc / cnt) * 1000.0),
        "learning_update_ms": float((t_upd / cnt) * 1000.0),
        "comm_overhead_bytes": int(n * 4 + 64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runtime_breakdown.csv")
    ap.add_argument("--swarm-sizes", default="20,50,100,200,500")
    ap.add_argument("--tasks", type=int, default=250)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sizes = [int(x.strip()) for x in args.swarm_sizes.split(",") if x.strip()]
    rows: List[Dict] = []

    for n in sizes:
        for m in METHODS:
            r = profile(m, n, args.tasks, args.seed, args.device)
            rows.append(r)
            print(f"[runtime] {r['method']:<10} n={n:>3} total={r['bid_compute_ms'] + r['auction_resolve_ms'] + r['learning_update_ms']:.3f} ms")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method", "swarm_size", "bid_compute_ms", "auction_resolve_ms", "learning_update_ms", "comm_overhead_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
