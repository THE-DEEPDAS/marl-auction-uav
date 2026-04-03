"""Run scenario sensitivity sweep (real implementation) and export scenario_sensitivity.csv."""

from __future__ import annotations

import argparse
import csv
import sys
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


def run_one(seed: int, arrival_rate: float, deadline_buffer_s: float, num_drones: int, tasks: int, device: str) -> Dict:
    sim = SwarmSimulator(
        num_drones=num_drones,
        task_arrival_rate=arrival_rate,
        duration=max(120.0, tasks / max(arrival_rate, 1e-6) + 5.0),
        seed=seed,
    )
    pool = AgentPool(
        num_drones,
        agent_type="daca",
        daca_config=DACAConfig(
            use_energy_awareness=True,
            device=device,
            learning_rate=0.015,
            critic_lr=0.04,
            behavior_lr=0.10,
            anchor_mix=0.60,
            model_mix=0.95,
        ),
    )

    for step in range(tasks):
        task = sim.next_task()
        if task is None:
            break

        # Override deadline buffer for stress condition
        task.deadline = float(deadline_buffer_s)

        obs = {d.drone_id: sim.get_observation(d, task) for d in sim.drones}
        eps = 0.02 * (0.9996 ** step)
        bids = pool.compute_bids(obs, exploration_noise=eps)
        result = sim.run_auction(task, bids)

        truthful = result.get("truthful_bids", {})
        rewards = result.get("rewards", {})
        winner = result.get("winner_id", None)

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

    st = sim.get_stats_summary()
    return {
        "method": "DACA",
        "arrival_rate": float(arrival_rate),
        "deadline_buffer_s": float(deadline_buffer_s),
        "seed": int(seed),
        "acceptance": float(st["task_acceptance_rate"] * 100.0),
        "welfare": float(st["normalized_welfare"]),
        "energy_kj_per_task": float(st["avg_energy_consumption"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="scenario_sensitivity.csv")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arrival-rates", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--deadline-buffers", default="60,120,180,240,300")
    ap.add_argument("--num-drones", type=int, default=200)
    ap.add_argument("--tasks", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    arrivals = [float(x.strip()) for x in args.arrival_rates.split(",") if x.strip()]
    deadlines = [float(x.strip()) for x in args.deadline_buffers.split(",") if x.strip()]

    rows: List[Dict] = []
    for lam in arrivals:
        for d in deadlines:
            for s in seeds:
                r = run_one(s, lam, d, args.num_drones, args.tasks, args.device)
                rows.append(r)
                print(f"[sensitivity] λ={lam:.2f}, d={d:.0f}, seed={s}, acc={r['acceptance']:.2f}%")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method", "arrival_rate", "deadline_buffer_s", "seed", "acceptance", "welfare", "energy_kj_per_task"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
