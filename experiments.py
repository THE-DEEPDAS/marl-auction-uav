"""Run reproducible experiments for auction-based UAV task allocation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

from agents import AgentPool, DACAConfig
from simulator import SwarmSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


DRONE_SWARM_SIZES = [20, 50, 100, 200]
METHODS = ["daca", "qlearning", "auction_nolearning", "greedy"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def _mean_std(values: List[float]) -> Dict[str, object]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "raw": []}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "raw": [float(x) for x in values],
    }


def _auction_outcome_from_profile(profile_bids: Dict[int, float], feasible_ids: List[int]) -> Tuple[int | None, float]:
    if not feasible_ids:
        return None, 0.0

    ranked = sorted(
        [(float(max(0.0, profile_bids.get(i, 0.0))), i) for i in feasible_ids],
        key=lambda x: x[0],
        reverse=True,
    )
    winner_bid, winner_id = ranked[0]
    payment = ranked[1][0] if len(ranked) > 1 else 0.0
    return winner_id, float(payment)


def _best_response_gain(
    bids: Dict[int, float],
    truthful: Dict[int, float],
    feasible_ids: List[int],
) -> float:
    winner_id, payment = _auction_outcome_from_profile(bids, feasible_ids)
    current_util = {i: 0.0 for i in feasible_ids}
    if winner_id is not None:
        current_util[winner_id] = float(max(0.0, truthful.get(winner_id, 0.0)) - payment)

    gains: List[float] = []
    for i in feasible_ids:
        dev_bids = dict(bids)
        dev_bids[i] = float(max(0.0, truthful.get(i, 0.0)))
        dev_winner, dev_payment = _auction_outcome_from_profile(dev_bids, feasible_ids)
        dev_util = 0.0
        if dev_winner == i:
            dev_util = float(max(0.0, truthful.get(i, 0.0)) - dev_payment)
        gains.append(dev_util - current_util[i])
    return float(max(0.0, max(gains) if gains else 0.0))


def _window_mean(arr: List[float], window_size: int) -> List[float]:
    if not arr or window_size <= 0:
        return []
    out: List[float] = []
    for i in range(0, len(arr), window_size):
        chunk = arr[i : i + window_size]
        out.append(float(np.mean(chunk)))
    return out


def _aggregate_windows(traces: List[List[float]]) -> Dict[str, List[float]]:
    if not traces:
        return {"mean": [], "std": []}
    min_len = min(len(t) for t in traces)
    if min_len == 0:
        return {"mean": [], "std": []}
    arr = np.array([t[:min_len] for t in traces], dtype=np.float64)
    return {
        "mean": [float(x) for x in np.mean(arr, axis=0)],
        "std": [float(x) for x in np.std(arr, axis=0)],
    }


def run_rollout(
    simulator: SwarmSimulator,
    agent_pool: AgentPool,
    max_tasks: int,
    exploration_noise_start: float = 0.06,
    exploration_noise_decay: float = 0.9995,
    learning_enabled: bool = True,
) -> Dict[str, object]:
    lyapunov_trace: List[float] = []
    rewards_trace: List[float] = []

    for step in range(max_tasks):
        task = simulator.next_task()
        if task is None:
            break

        obs: Dict[int, np.ndarray] = {
            drone.drone_id: simulator.get_observation(drone, task)
            for drone in simulator.drones
        }

        eps = exploration_noise_start * (exploration_noise_decay ** step)
        bids = agent_pool.compute_bids(obs, exploration_noise=eps)
        result = simulator.run_auction(task, bids)

        lyapunov_trace.append(float(result["lyapunov_distance"]))
        rewards = result["rewards"]
        rewards_trace.append(float(np.mean(list(rewards.values()))))

        if learning_enabled:
            truthful = result.get("truthful_bids", {})
            winner_id = result.get("winner_id", None)
            for drone_id, state in obs.items():
                next_state = state
                reward = rewards.get(drone_id, 0.0)

                # Team-aware shaping: all agents observe completion signal,
                # winner keeps dense utility while non-winners get small coordination reward.
                if winner_id is not None:
                    if drone_id == winner_id:
                        reward += 0.10 * float(task.priority / 100.0)
                    else:
                        reward += 0.02 * float(task.priority / 100.0)

                agent_pool.update_agent(
                    drone_id,
                    state,
                    bids.get(drone_id, 0.0),
                    reward,
                    next_state,
                    False,
                    truthful_bid=float(truthful.get(drone_id, 0.0)),
                    winner_id=winner_id,
                )

    return {
        "stats": simulator.get_stats_summary(),
        "lyapunov_trace": lyapunov_trace,
        "mean_reward_trace": rewards_trace,
    }


def run_method_comparison(
    swarm_sizes: Iterable[int],
    seeds: Iterable[int],
    duration: float = 600.0,
    task_arrival_rate: float = 1.0,
    output_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, object]:
    all_results: Dict[str, object] = {}
    os.makedirs(output_dir, exist_ok=True)

    for method in METHODS:
        logger.info("Method comparison: %s", method)
        by_size: Dict[str, object] = {}

        for n in swarm_sizes:
            metrics = {
                "task_acceptance": [],
                "avg_energy": [],
                "social_welfare": [],
                "fairness": [],
                "lyapunov_last": [],
            }

            for seed in seeds:
                simulator = SwarmSimulator(
                    num_drones=n,
                    task_arrival_rate=task_arrival_rate,
                    duration=duration,
                    seed=seed,
                )

                daca_cfg = DACAConfig(
                    use_energy_awareness=True,
                    device=device,
                    learning_rate=0.015,
                    critic_lr=0.04,
                    behavior_lr=0.10,
                    anchor_mix=0.60,
                    model_mix=0.95,
                )
                pool = AgentPool(n, agent_type=method, daca_config=daca_cfg)
                rollout = run_rollout(
                    simulator,
                    pool,
                    max_tasks=int(duration * task_arrival_rate * 1.2),
                    exploration_noise_start=0.02 if method == "daca" else (0.10 if method == "qlearning" else 0.0),
                    exploration_noise_decay=0.9996 if method == "daca" else 0.999,
                    learning_enabled=method in ("daca", "qlearning"),
                )

                stats = rollout["stats"]
                metrics["task_acceptance"].append(stats["task_acceptance_rate"] * 100.0)
                metrics["avg_energy"].append(stats["avg_energy_consumption"])
                metrics["social_welfare"].append(stats["normalized_welfare"])
                metrics["fairness"].append(stats["fairness_index"])
                lyap = rollout["lyapunov_trace"]
                metrics["lyapunov_last"].append(float(lyap[-1]) if lyap else 0.0)

            by_size[str(n)] = {
                "task_acceptance": _mean_std(metrics["task_acceptance"]),
                "avg_energy": _mean_std(metrics["avg_energy"]),
                "social_welfare": _mean_std(metrics["social_welfare"]),
                "fairness": _mean_std(metrics["fairness"]),
                "lyapunov_last": _mean_std(metrics["lyapunov_last"]),
            }

        method_result = {
            "method": method,
            "swarm_sizes": list(swarm_sizes),
            "results_by_size": by_size,
            "seeds": list(seeds),
        }
        all_results[f"method_comparison_{method}"] = method_result

        with open(os.path.join(output_dir, f"method_comparison_{method}.json"), "w", encoding="utf-8") as f:
            json.dump(method_result, f, indent=2)

    return all_results


def run_convergence_experiment(
    seeds: Iterable[int],
    num_tasks_grid: Iterable[int] = (100, 300, 1000, 3000, 10000),
    output_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, object] = {}

    for method in METHODS:
        means: List[float] = []
        stds: List[float] = []
        n_list = list(num_tasks_grid)

        for n_tasks in n_list:
            seed_vals: List[float] = []
            for seed in seeds:
                simulator = SwarmSimulator(
                    num_drones=100,
                    task_arrival_rate=10.0,
                    duration=max(120.0, n_tasks / 10.0 + 1.0),
                    seed=seed,
                )
                daca_cfg = DACAConfig(device=device, learning_rate=0.015, critic_lr=0.04, behavior_lr=0.10, anchor_mix=0.60, model_mix=0.95)
                pool = AgentPool(100, agent_type=method, daca_config=daca_cfg)
                rollout = run_rollout(
                    simulator,
                    pool,
                    max_tasks=n_tasks,
                    exploration_noise_start=0.02 if method == "daca" else (0.10 if method == "qlearning" else 0.0),
                    exploration_noise_decay=0.9995,
                    learning_enabled=method in ("daca", "qlearning"),
                )
                lyap = rollout["lyapunov_trace"]
                seed_vals.append(float(np.mean(lyap[-50:])) if len(lyap) >= 50 else float(np.mean(lyap) if lyap else 0.0))

            means.append(float(np.mean(seed_vals)))
            stds.append(float(np.std(seed_vals)))

        results[method] = {
            "num_tasks": n_list,
            "convergence_distances": means,
            "convergence_std": stds,
        }

    with open(os.path.join(output_dir, "convergence.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def run_real_convergence_experiment(
    seeds: Iterable[int],
    total_tasks: int = 6000,
    window_size: int = 250,
    output_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    result: Dict[str, object] = {
        "meta": {
            "num_drones": 100,
            "total_tasks": int(total_tasks),
            "window_size": int(window_size),
            "seeds": [int(s) for s in seeds],
        },
        "methods": {},
    }

    for method in METHODS:
        lyap_traces: List[List[float]] = []
        br_gain_traces: List[List[float]] = []
        bid_mae_traces: List[List[float]] = []
        eff_gap_traces: List[List[float]] = []

        for seed in seeds:
            simulator = SwarmSimulator(
                num_drones=100,
                task_arrival_rate=10.0,
                duration=max(120.0, total_tasks / 10.0 + 5.0),
                seed=int(seed),
            )

            daca_cfg = DACAConfig(
                use_energy_awareness=True,
                device=device,
                learning_rate=0.015,
                critic_lr=0.04,
                behavior_lr=0.10,
                anchor_mix=0.60,
                model_mix=0.95,
            )
            pool = AgentPool(100, agent_type=method, daca_config=daca_cfg)

            lyap_trace: List[float] = []
            br_gain_trace: List[float] = []
            bid_mae_trace: List[float] = []
            eff_gap_trace: List[float] = []

            for step in range(total_tasks):
                task = simulator.next_task()
                if task is None:
                    break

                obs: Dict[int, np.ndarray] = {
                    drone.drone_id: simulator.get_observation(drone, task)
                    for drone in simulator.drones
                }

                eps = 0.02 * (0.9996 ** step) if method == "daca" else (0.10 * (0.999 ** step) if method == "qlearning" else 0.0)
                bids = pool.compute_bids(obs, exploration_noise=eps)

                feasible_ids = [d.drone_id for d in simulator.drones if simulator.compute_feasibility(d, task)]
                truthful = {i: float(max(0.0, simulator.compute_valuation(simulator.drones[i], task))) for i in feasible_ids}

                result_step = simulator.run_auction(task, bids)
                winner_id = result_step.get("winner_id", None)

                # Convergence signals: truthful calibration, unilateral deviation gain, and efficiency loss.
                lyap_trace.append(float(result_step.get("lyapunov_distance", 0.0)))
                br_gain_trace.append(_best_response_gain(bids, truthful, feasible_ids))

                if feasible_ids:
                    mae = [abs(float(max(0.0, bids.get(i, 0.0))) - truthful.get(i, 0.0)) for i in feasible_ids]
                    bid_mae_trace.append(float(np.mean(mae)))

                    best_truth = max(truthful.values()) if truthful else 0.0
                    winner_truth = truthful.get(winner_id, 0.0) if winner_id is not None else 0.0
                    eff_gap_trace.append(float(max(0.0, best_truth - winner_truth)))
                else:
                    bid_mae_trace.append(0.0)
                    eff_gap_trace.append(0.0)

                truthful_step = result_step.get("truthful_bids", {})
                rewards = result_step.get("rewards", {})
                winner = result_step.get("winner_id", None)
                if method in ("daca", "qlearning"):
                    for drone_id, state in obs.items():
                        reward = float(rewards.get(drone_id, 0.0))
                        if winner is not None:
                            if drone_id == winner:
                                reward += 0.10 * float(task.priority / 100.0)
                            else:
                                reward += 0.02 * float(task.priority / 100.0)
                        pool.update_agent(
                            drone_id,
                            state,
                            float(max(0.0, bids.get(drone_id, 0.0))),
                            reward,
                            state,
                            False,
                            truthful_bid=float(truthful_step.get(drone_id, 0.0)),
                            winner_id=winner,
                        )

            lyap_traces.append(_window_mean(lyap_trace, window_size))
            br_gain_traces.append(_window_mean(br_gain_trace, window_size))
            bid_mae_traces.append(_window_mean(bid_mae_trace, window_size))
            eff_gap_traces.append(_window_mean(eff_gap_trace, window_size))

        agg_lyap = _aggregate_windows(lyap_traces)
        agg_br = _aggregate_windows(br_gain_traces)
        agg_mae = _aggregate_windows(bid_mae_traces)
        agg_eff = _aggregate_windows(eff_gap_traces)

        n_windows = len(agg_lyap["mean"])
        window_midpoints = [int((i + 0.5) * window_size) for i in range(n_windows)]

        result["methods"][method] = {
            "window_midpoints": window_midpoints,
            "lyapunov_mean": agg_lyap["mean"],
            "lyapunov_std": agg_lyap["std"],
            "best_response_gain_mean": agg_br["mean"],
            "best_response_gain_std": agg_br["std"],
            "bid_mae_mean": agg_mae["mean"],
            "bid_mae_std": agg_mae["std"],
            "efficiency_gap_mean": agg_eff["mean"],
            "efficiency_gap_std": agg_eff["std"],
        }

    with open(os.path.join(output_dir, "real_convergence.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_scalability_experiment(
    swarm_sizes: Iterable[int] = (20, 50, 100, 200, 500),
    output_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    out: Dict[str, List[Dict[str, float]]] = {m: [] for m in METHODS}

    for n in swarm_sizes:
        for method in METHODS:
            simulator = SwarmSimulator(num_drones=n, task_arrival_rate=2.0, duration=200.0, seed=77)
            daca_cfg = DACAConfig(device=device)
            pool = AgentPool(n, method, daca_config=daca_cfg)
            times: List[float] = []

            for _ in range(200):
                task = simulator.next_task()
                if task is None:
                    break
                obs = {d.drone_id: simulator.get_observation(d, task) for d in simulator.drones}
                t0 = time.perf_counter()
                bids = pool.compute_bids(obs, exploration_noise=0.0)
                simulator.run_auction(task, bids)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            out[method].append(
                {
                    "swarm_size": float(n),
                    "avg_time_ms": float(np.mean(times) if times else 0.0),
                    "std_time_ms": float(np.std(times) if times else 0.0),
                }
            )

    with open(os.path.join(output_dir, "scalability.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_ablation_study(
    seeds: Iterable[int],
    output_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    variants = {
        "full_daca": {"agent_type": "daca", "daca_cfg": DACAConfig(use_energy_awareness=True, device=device)},
        "without_energy_awareness": {"agent_type": "daca", "daca_cfg": DACAConfig(use_energy_awareness=False, device=device)},
        "without_auction_mechanism": {"agent_type": "qlearning", "daca_cfg": DACAConfig(device=device)},
        "without_policy_gradient": {"agent_type": "auction_nolearning", "daca_cfg": DACAConfig(device=device)},
        "without_valuation_learning": {"agent_type": "greedy", "daca_cfg": DACAConfig(device=device)},
    }

    results: Dict[str, float] = {}
    for name, conf in variants.items():
        vals: List[float] = []
        for seed in seeds:
            simulator = SwarmSimulator(num_drones=100, task_arrival_rate=1.0, duration=600.0, seed=seed)
            pool = AgentPool(100, conf["agent_type"], daca_config=conf["daca_cfg"])
            rollout = run_rollout(
                simulator,
                pool,
                max_tasks=800,
                exploration_noise_start=0.02 if conf["agent_type"] == "daca" else (0.10 if conf["agent_type"] == "qlearning" else 0.0),
                exploration_noise_decay=0.999,
                learning_enabled=conf["agent_type"] in ("daca", "qlearning"),
            )
            vals.append(rollout["stats"]["task_acceptance_rate"] * 100.0)
        results[name] = float(np.mean(vals))

    out = {
        "swarm_size": 100,
        "metric": "task_acceptance_rate",
        "results": results,
    }
    with open(os.path.join(output_dir, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_all(output_dir: str = "results", seeds: Iterable[int] = DEFAULT_SEEDS, device: str = "cpu") -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    all_results: Dict[str, object] = {}
    all_results.update(run_method_comparison(DRONE_SWARM_SIZES, seeds, output_dir=output_dir, device=device))
    all_results["convergence"] = run_convergence_experiment(seeds, output_dir=output_dir, device=device)
    all_results["real_convergence"] = run_real_convergence_experiment(seeds, output_dir=output_dir, device=device)
    all_results["scalability"] = run_scalability_experiment(output_dir=output_dir, device=device)
    all_results["ablation"] = run_ablation_study(seeds, output_dir=output_dir, device=device)

    with open(os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auction-based UAV task allocation experiments")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "method", "convergence", "real_convergence", "scalability", "ablation"],
        help="Experiment suite to run",
    )
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for JSON outputs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=DEFAULT_SEEDS,
        help="Random seeds to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for DACA updates: cpu or cuda",
    )
    parser.add_argument("--total_tasks", type=int, default=6000, help="Total tasks for real_convergence suite")
    parser.add_argument("--window_size", type=int, default=250, help="Window size for real_convergence metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device.startswith("cuda"):
        logger.info("CUDA requested; DACA learning updates will use GPU when torch+CUDA is available.")
        logger.info("Simulator/event loop remains CPU-bound to preserve exact environment dynamics.")

    if args.suite == "all":
        run_all(args.output_dir, args.seeds, device=args.device)
    elif args.suite == "method":
        run_method_comparison(DRONE_SWARM_SIZES, args.seeds, output_dir=args.output_dir, device=args.device)
    elif args.suite == "convergence":
        run_convergence_experiment(args.seeds, output_dir=args.output_dir, device=args.device)
    elif args.suite == "real_convergence":
        run_real_convergence_experiment(
            args.seeds,
            total_tasks=args.total_tasks,
            window_size=args.window_size,
            output_dir=args.output_dir,
            device=args.device,
        )
    elif args.suite == "scalability":
        run_scalability_experiment(output_dir=args.output_dir, device=args.device)
    elif args.suite == "ablation":
        run_ablation_study(args.seeds, output_dir=args.output_dir, device=args.device)

    logger.info("Completed suite: %s", args.suite)


if __name__ == "__main__":
    main()
