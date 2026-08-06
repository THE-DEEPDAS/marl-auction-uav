"""Run corrected workload/urgency stress tests for the final manuscript."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents import AgentPool, DACAConfig
from src.experiments import run_rollout
from src.simulator import SwarmSimulator

OUT = Path("results/final_stress_cuda")
METHODS = ["daca", "truthful_value", "auction_nolearning"]
SCENARIOS = [("sparse", 0.5, 60.0), ("nominal", 1.0, 60.0), ("loaded", 2.0, 30.0)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, rate, duration in SCENARIOS:
        result[name] = {}
        for method in METHODS:
            rows = []
            for seed in range(5):
                sim = SwarmSimulator(num_drones=100, task_arrival_rate=rate, duration=duration, seed=seed)
                pool = AgentPool(100, method, daca_config=DACAConfig(
                    device="cuda", anchor_mix=1.0, model_mix=1.0))
                rollout = run_rollout(sim, pool, max_tasks=int(duration * rate * 1.2),
                                      exploration_noise_start=0.0 if method == "daca" else (0.10 if method == "qlearning" else 0.0),
                                      exploration_noise_decay=0.9996 if method == "daca" else 0.999,
                                      learning_enabled=method in ("daca", "qlearning"))
                rows.append(rollout["stats"])
            keys = ["task_acceptance_rate", "normalized_welfare", "avg_energy_consumption", "allocation_efficiency", "allocation_regret"]
            result[name][method] = {k: {"mean": sum(r[k] for r in rows)/len(rows), "raw": [r[k] for r in rows]} for k in keys}
        (OUT / f"{name}.json").write_text(json.dumps(result[name], indent=2), encoding="utf-8")
    (OUT / "stress_matrix.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT / "stress_matrix.json")

if __name__ == "__main__":
    main()
