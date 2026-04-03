"""
CUDA-enabled experiment runner for UAV auction allocation.

This script produces the CSV artifacts used by:
  - analysis/generate_additional_figures.py

Outputs (in --output-dir):
  - seed_metrics.csv
  - scenario_sensitivity.csv
  - runtime_breakdown.csv

Usage example (PowerShell):
  python analysis/run_experiments_cuda.py --output-dir results --require-cuda

Notes:
  - DACA is trained/inferred with PyTorch on CUDA when available.
  - AL, Q-learning, and Greedy are CPU baselines.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# Configuration and data types
# ----------------------------

@dataclass
class UAVConfig:
    speed: float         # m/s
    energy_rate: float   # J/m
    e_max: float         # kJ


UAV_TYPES = {
    0: UAVConfig(speed=20.0, energy_rate=50.0, e_max=50.0),
    1: UAVConfig(speed=15.0, energy_rate=35.0, e_max=40.0),
    2: UAVConfig(speed=10.0, energy_rate=20.0, e_max=30.0),
}


@dataclass
class ExperimentConfig:
    area_size_m: float = 5000.0
    process_time_s: float = 20.0
    priority_min: float = 10.0
    priority_max: float = 100.0
    deadline_buffer_min_s: float = 60.0
    deadline_buffer_max_s: float = 300.0

    # Valuation weights
    alpha_energy: float = 0.55
    beta_load: float = 3.0
    gamma_travel: float = 0.2

    # Auction bounds
    b_max: float = 100.0

    # Dynamics
    load_max: int = 10
    reset_energy_each_task: bool = False


# ----------------------------
# Utility functions
# ----------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    lo = np.min(v)
    hi = np.max(v)
    return (v - lo) / (hi - lo + eps)


# ----------------------------
# Environment and features
# ----------------------------


class SwarmState:
    def __init__(self, n_agents: int, cfg: ExperimentConfig, rng: np.random.Generator):
        self.n = n_agents
        self.cfg = cfg
        self.rng = rng

        # Type composition: 30/40/30 split
        n_a = int(round(0.30 * n_agents))
        n_b = int(round(0.40 * n_agents))
        n_c = n_agents - n_a - n_b
        types = np.array([0] * n_a + [1] * n_b + [2] * n_c, dtype=np.int64)
        rng.shuffle(types)
        self.types = types

        self.pos = rng.uniform(0, cfg.area_size_m, size=(n_agents, 2)).astype(np.float32)
        self.energy_kj = np.array([UAV_TYPES[t].e_max for t in types], dtype=np.float32)
        self.load = np.zeros(n_agents, dtype=np.int32)

        self.completed = np.zeros(n_agents, dtype=np.int32)
        self.energy_used_kj = np.zeros(n_agents, dtype=np.float64)

        self.t = 0.0

    def speeds(self) -> np.ndarray:
        return np.array([UAV_TYPES[t].speed for t in self.types], dtype=np.float32)

    def energy_rates(self) -> np.ndarray:
        return np.array([UAV_TYPES[t].energy_rate for t in self.types], dtype=np.float32)


@dataclass
class Task:
    x: float
    y: float
    priority: float
    deadline_buffer_s: float



def sample_task(cfg: ExperimentConfig, rng: np.random.Generator, deadline_override: float | None = None) -> Task:
    x, y = rng.uniform(0, cfg.area_size_m, size=2)
    p = rng.uniform(cfg.priority_min, cfg.priority_max)
    if deadline_override is None:
        d = rng.uniform(cfg.deadline_buffer_min_s, cfg.deadline_buffer_max_s)
    else:
        d = float(deadline_override)
    return Task(float(x), float(y), float(p), d)


def compute_features_and_valuation(
    swarm: SwarmState,
    task: Task,
    cfg: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = swarm.pos
    tx = np.array([task.x, task.y], dtype=np.float32)
    dists = np.linalg.norm(pos - tx[None, :], axis=1).astype(np.float32)

    speeds = swarm.speeds()
    c_rates = swarm.energy_rates()  # J/m

    travel_t = dists / np.maximum(speeds, 1e-6)
    total_t = travel_t + cfg.process_time_s

    # round trip energy in kJ
    energy_kj = (2.0 * c_rates * dists) / 1000.0

    feasible_deadline = total_t <= task.deadline_buffer_s
    feasible_energy = energy_kj <= swarm.energy_kj
    feasible = feasible_deadline & feasible_energy

    valuation = (
        task.priority
        - cfg.alpha_energy * energy_kj
        - cfg.beta_load * swarm.load.astype(np.float32)
        - cfg.gamma_travel * travel_t
    ).astype(np.float32)

    # Features in [0,1]-like ranges
    f_dist = np.clip(dists / cfg.area_size_m, 0.0, 1.0)
    f_energy = np.clip(swarm.energy_kj / np.array([UAV_TYPES[t].e_max for t in swarm.types], dtype=np.float32), 0.0, 1.0)
    f_load = np.clip(swarm.load / max(1, cfg.load_max), 0.0, 1.0).astype(np.float32)
    f_speed = speeds / 20.0
    f_rate = c_rates / 50.0
    f_priority = np.full(swarm.n, task.priority / cfg.priority_max, dtype=np.float32)
    f_deadline = np.full(swarm.n, task.deadline_buffer_s / cfg.deadline_buffer_max_s, dtype=np.float32)

    features = np.stack([f_dist, f_energy, f_load, f_speed, f_rate, f_priority, f_deadline], axis=1).astype(np.float32)

    return features, valuation, feasible, dists, travel_t, energy_kj


# ----------------------------
# Baseline policies
# ----------------------------


class ALPolicy:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def bid(self, valuation: np.ndarray, feasible: np.ndarray, _features: np.ndarray) -> np.ndarray:
        b = np.clip(valuation, 0.0, self.cfg.b_max)
        b[~feasible] = 0.0
        return b


class GreedyPolicy:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def bid(self, valuation: np.ndarray, feasible: np.ndarray, features: np.ndarray) -> np.ndarray:
        # Priority-heavy heuristic with energy/load penalties from features
        dist = features[:, 0]
        energy_ratio = 1.0 - features[:, 1]
        load = features[:, 2]
        priority = features[:, 5] * self.cfg.priority_max

        b = priority - 20.0 * dist - 15.0 * energy_ratio - 10.0 * load
        b = np.clip(b, 0.0, self.cfg.b_max)
        b[~feasible] = 0.0
        return b.astype(np.float32)


class QLearningPolicy:
    def __init__(self, cfg: ExperimentConfig, n_actions: int = 11):
        self.cfg = cfg
        self.n_actions = n_actions
        self.q: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self.alpha = 0.15
        self.gamma = 0.95
        self.eps = 0.15

    def _discretize(self, features: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # (dist, energy, load, priority)
        bins = []
        for f in features:
            dist_b = int(np.clip(f[0] * 4, 0, 3))
            energy_b = int(np.clip(f[1] * 4, 0, 3))
            load_b = int(np.clip(f[2] * 4, 0, 3))
            p_b = int(np.clip(f[5] * 4, 0, 3))
            bins.append((dist_b, energy_b, load_b, p_b))
        return bins

    def _ensure(self, key: Tuple[int, int, int, int]) -> np.ndarray:
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q[key]

    def bid(self, valuation: np.ndarray, feasible: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, List[int], List[Tuple[int, int, int, int]]]:
        keys = self._discretize(features)
        bids = np.zeros(features.shape[0], dtype=np.float32)
        acts = []

        for i, k in enumerate(keys):
            qv = self._ensure(k)
            if np.random.rand() < self.eps:
                a = np.random.randint(0, self.n_actions)
            else:
                a = int(np.argmax(qv))
            frac = a / (self.n_actions - 1)
            bids[i] = np.clip(frac * self.cfg.b_max, 0.0, self.cfg.b_max)
            acts.append(a)

        bids[~feasible] = 0.0
        return bids, acts, keys

    def update(self, key: Tuple[int, int, int, int], action: int, reward: float, next_key: Tuple[int, int, int, int]) -> None:
        q = self._ensure(key)
        q_next = self._ensure(next_key)
        td_target = reward + self.gamma * float(np.max(q_next))
        q[action] = q[action] + self.alpha * (td_target - q[action])


# ----------------------------
# DACA (CUDA)
# ----------------------------


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int = 7, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.actor_mu = nn.Linear(hidden // 2, 1)
        self.actor_logstd = nn.Parameter(torch.tensor(-1.5))
        self.critic = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = torch.sigmoid(self.actor_mu(h)).squeeze(-1)
        std = torch.exp(self.actor_logstd).clamp(1e-3, 2.0)
        v = self.critic(h).squeeze(-1)
        return mu, std, v


class DACAPolicy:
    def __init__(self, cfg: ExperimentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.net = ActorCritic().to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=1.5e-3)

        self.gamma = 0.99
        self.entropy_coef = 1e-3
        self.value_coef = 0.5

        self.storage: List[Dict[str, torch.Tensor]] = []
        self.update_every = 32

    def bid(self, features: np.ndarray, feasible: np.ndarray) -> Tuple[np.ndarray, Dict[str, torch.Tensor], float]:
        t0 = time.perf_counter()
        x = torch.from_numpy(features).to(self.device)
        mu, std, v = self.net(x)

        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        bid = torch.clamp(raw * self.cfg.b_max, 0.0, self.cfg.b_max)

        if feasible is not None:
            mask = torch.from_numpy(feasible.astype(np.bool_)).to(self.device)
            bid = torch.where(mask, bid, torch.zeros_like(bid))

        out = {
            "x": x,
            "mu": mu,
            "std": std,
            "v": v,
            "bid": bid,
            "logp": dist.log_prob(raw),
            "entropy": dist.entropy(),
        }
        t1 = time.perf_counter()
        return bid.detach().cpu().numpy().astype(np.float32), out, (t1 - t0) * 1000.0

    def record_step(self, out: Dict[str, torch.Tensor], rewards: np.ndarray) -> None:
        r = torch.from_numpy(rewards.astype(np.float32)).to(self.device)
        self.storage.append({
            "logp": out["logp"],
            "value": out["v"],
            "reward": r,
            "entropy": out["entropy"],
        })

    def maybe_update(self) -> float:
        if len(self.storage) < self.update_every:
            return 0.0

        t0 = time.perf_counter()

        rewards = torch.stack([s["reward"] for s in self.storage], dim=0)      # [T, N]
        values = torch.stack([s["value"] for s in self.storage], dim=0)        # [T, N]
        logps = torch.stack([s["logp"] for s in self.storage], dim=0)          # [T, N]
        ents = torch.stack([s["entropy"] for s in self.storage], dim=0)        # [T, N]

        # discounted returns per agent
        T, N = rewards.shape
        returns = torch.zeros_like(rewards)
        g = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            g = rewards[t] + self.gamma * g
            returns[t] = g

        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        actor_loss = -(logps * adv.detach()).mean()
        critic_loss = (returns.detach() - values).pow(2).mean()
        ent_loss = -ents.mean()

        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * ent_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        self.storage.clear()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0


# ----------------------------
# Core simulation loop
# ----------------------------


def step_auction(
    bids: np.ndarray,
    valuation: np.ndarray,
    feasible: np.ndarray,
    energy_kj: np.ndarray,
) -> Tuple[int, float, np.ndarray, bool]:
    # winner among feasible positive bids
    if np.all(~feasible) or np.max(bids) <= 0:
        return -1, 0.0, np.zeros_like(bids, dtype=np.float32), False

    winner = int(np.argmax(bids))
    sorted_bids = np.sort(bids)
    payment = float(sorted_bids[-2]) if bids.size >= 2 else 0.0

    rewards = np.zeros_like(bids, dtype=np.float32)
    rewards[winner] = float(valuation[winner] - payment)

    accepted = bool(feasible[winner] and valuation[winner] > 0.0)
    return winner, payment, rewards, accepted


def run_method_once(
    method: str,
    n_agents: int,
    n_tasks: int,
    seed: int,
    cfg: ExperimentConfig,
    arrival_rate: float,
    deadline_override: float | None,
    device: torch.device,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    swarm = SwarmState(n_agents=n_agents, cfg=cfg, rng=rng)

    al = ALPolicy(cfg)
    greedy = GreedyPolicy(cfg)
    ql = QLearningPolicy(cfg)
    daca = DACAPolicy(cfg, device=device) if method == "DACA" else None

    accepted = 0
    total = 0
    total_priority = 0.0
    total_possible_priority = 0.0
    completed_energy = []

    latency_ms_total = 0.0
    bid_compute_ms_total = 0.0
    auction_resolve_ms_total = 0.0
    learning_update_ms_total = 0.0

    for t in range(n_tasks):
        # Poisson arrivals via exponential inter-arrival times
        inter = rng.exponential(1.0 / max(arrival_rate, 1e-6))
        swarm.t += float(inter)

        task = sample_task(cfg, rng, deadline_override=deadline_override)
        features, valuation, feasible, _d, _tt, energy_kj = compute_features_and_valuation(swarm, task, cfg)

        t0 = time.perf_counter()
        if method == "AL":
            bids = al.bid(valuation, feasible, features)
            bid_compute_ms = (time.perf_counter() - t0) * 1000.0
            out = None
        elif method == "Greedy":
            bids = greedy.bid(valuation, feasible, features)
            bid_compute_ms = (time.perf_counter() - t0) * 1000.0
            out = None
        elif method == "Q-learning":
            bids, acts, keys = ql.bid(valuation, feasible, features)
            bid_compute_ms = (time.perf_counter() - t0) * 1000.0
            out = (acts, keys)
        elif method == "DACA":
            bids, out, bid_compute_ms = daca.bid(features, feasible)
        else:
            raise ValueError(f"Unknown method: {method}")

        t1 = time.perf_counter()
        winner, payment, rewards, is_accepted = step_auction(bids, valuation, feasible, energy_kj)
        auction_resolve_ms = (time.perf_counter() - t1) * 1000.0

        # Environment transition and stats
        total += 1
        total_possible_priority += task.priority
        if winner >= 0:
            # resource updates
            swarm.energy_kj[winner] = max(0.0, float(swarm.energy_kj[winner] - energy_kj[winner]))
            swarm.energy_used_kj[winner] += float(energy_kj[winner])
            swarm.completed[winner] += 1
            swarm.load[winner] = min(cfg.load_max, swarm.load[winner] + 1)
            swarm.pos[winner, 0] = task.x
            swarm.pos[winner, 1] = task.y

            if is_accepted:
                accepted += 1
                total_priority += task.priority
                completed_energy.append(float(energy_kj[winner]))

            # Q-learning update for winner only
            if method == "Q-learning" and out is not None:
                acts, keys = out
                next_task = sample_task(cfg, rng, deadline_override=deadline_override)
                nxt_f, *_ = compute_features_and_valuation(swarm, next_task, cfg)
                next_key = ql._discretize(nxt_f)[winner]
                ql.update(keys[winner], acts[winner], float(rewards[winner]), next_key)

            if method == "DACA" and daca is not None and out is not None:
                daca.record_step(out, rewards)
        else:
            if method == "DACA" and daca is not None and out is not None:
                daca.record_step(out, rewards)

        learn_ms = 0.0
        if method == "DACA" and daca is not None:
            learn_ms = daca.maybe_update()

        # mild load decay to mimic queue service
        swarm.load = np.maximum(0, swarm.load - 1)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latency_ms_total += latency_ms
        bid_compute_ms_total += bid_compute_ms
        auction_resolve_ms_total += auction_resolve_ms
        learning_update_ms_total += learn_ms

    acceptance = 100.0 * accepted / max(1, total)
    welfare = total_priority / max(1e-6, total_possible_priority)
    energy_mean = float(np.mean(completed_energy)) if completed_energy else np.nan
    fairness = gini(swarm.completed.astype(np.float64))
    latency = latency_ms_total / max(1, n_tasks)

    return {
        "method": method,
        "swarm_size": n_agents,
        "seed": seed,
        "acceptance": acceptance,
        "welfare": welfare,
        "energy_kj_per_task": energy_mean,
        "fairness_gini": fairness,
        "latency_ms": latency,
        "bid_compute_ms": bid_compute_ms_total / max(1, n_tasks),
        "auction_resolve_ms": auction_resolve_ms_total / max(1, n_tasks),
        "learning_update_ms": learning_update_ms_total / max(1, n_tasks),
    }


# ----------------------------
# Experiment suites
# ----------------------------


def run_seed_suite(
    methods: List[str],
    swarm_sizes: List[int],
    seeds: List[int],
    tasks_per_run: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for n in swarm_sizes:
        for seed in seeds:
            set_seed(seed)
            for m in methods:
                row = run_method_once(
                    method=m,
                    n_agents=n,
                    n_tasks=tasks_per_run,
                    seed=seed,
                    cfg=cfg,
                    arrival_rate=1.0,
                    deadline_override=None,
                    device=device,
                )
                rows.append(row)
                print(f"[seed-suite] n={n:>3} seed={seed} method={m:<10} acc={row['acceptance']:.2f}%")
    return pd.DataFrame(rows)


def run_sensitivity_suite(
    method: str,
    n_agents: int,
    seeds: List[int],
    arrival_rates: List[float],
    deadline_buffers: List[float],
    tasks_per_cell: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for lam in arrival_rates:
        for d in deadline_buffers:
            for seed in seeds:
                set_seed(seed)
                row = run_method_once(
                    method=method,
                    n_agents=n_agents,
                    n_tasks=tasks_per_cell,
                    seed=seed,
                    cfg=cfg,
                    arrival_rate=lam,
                    deadline_override=d,
                    device=device,
                )
                rows.append({
                    "method": method,
                    "arrival_rate": lam,
                    "deadline_buffer_s": d,
                    "seed": seed,
                    "acceptance": row["acceptance"],
                    "welfare": row["welfare"],
                    "energy_kj_per_task": row["energy_kj_per_task"],
                })
                print(f"[sensitivity] λ={lam:.2f} d={d:>5.0f}s seed={seed} acc={row['acceptance']:.2f}%")
    return pd.DataFrame(rows)


def build_runtime_breakdown(seed_df: pd.DataFrame) -> pd.DataFrame:
    g = (
        seed_df.groupby(["method", "swarm_size"], as_index=False)[
            ["bid_compute_ms", "auction_resolve_ms", "learning_update_ms"]
        ]
        .mean()
        .sort_values(["method", "swarm_size"])
    )

    # approximate communication overhead for one-shot broadcast + winner response
    # float32 bid per agent + small header; in bytes
    g["comm_overhead_bytes"] = g["swarm_size"].astype(np.int64) * 4 + 64
    return g


# ----------------------------
# Main
# ----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run UAV auction experiments with CUDA-enabled DACA.")
    p.add_argument("--output-dir", default="results", type=str)
    p.add_argument("--tasks-per-run", default=1200, type=int)
    p.add_argument("--tasks-per-sensitivity-cell", default=500, type=int)
    p.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable.")
    p.add_argument("--seeds", default="0,1,2,3,4", type=str)
    p.add_argument("--swarm-sizes", default="20,50,100,200,500", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required (--require-cuda) but no CUDA device is available.")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    swarm_sizes = [int(x.strip()) for x in args.swarm_sizes.split(",") if x.strip()]

    cfg = ExperimentConfig()
    methods = ["AL", "Q-learning", "Greedy", "DACA"]

    seed_df = run_seed_suite(
        methods=methods,
        swarm_sizes=swarm_sizes,
        seeds=seeds,
        tasks_per_run=args.tasks_per_run,
        cfg=cfg,
        device=device,
    )
    seed_df.to_csv(out_dir / "seed_metrics.csv", index=False)

    sens_df = run_sensitivity_suite(
        method="DACA",
        n_agents=200,
        seeds=seeds,
        arrival_rates=[0.5, 1.0, 1.5, 2.0],
        deadline_buffers=[60, 120, 180, 240, 300],
        tasks_per_cell=args.tasks_per_sensitivity_cell,
        cfg=cfg,
        device=device,
    )
    sens_df.to_csv(out_dir / "scenario_sensitivity.csv", index=False)

    runtime_df = build_runtime_breakdown(seed_df)
    runtime_df.to_csv(out_dir / "runtime_breakdown.csv", index=False)

    print("\nSaved:")
    print(f"  - {out_dir / 'seed_metrics.csv'}")
    print(f"  - {out_dir / 'scenario_sensitivity.csv'}")
    print(f"  - {out_dir / 'runtime_breakdown.csv'}")


if __name__ == "__main__":
    main()
