"""Build seed_metrics.csv from real implementation JSON outputs.

Input: results directory containing method_comparison_*.json and optionally scalability.json
Output: seed_metrics.csv (for generate_additional_figures.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

METHOD_MAP = {
    "daca": "DACA",
    "qlearning": "Q-learning",
    "auction_nolearning": "AL",
    "greedy": "Greedy",
}


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", type=str)
    ap.add_argument("--output", default="seed_metrics.csv", type=str)
    args = ap.parse_args()

    root = Path(args.results_dir)
    scalability = load_json(root / "scalability.json")

    lat_lookup: Dict[tuple, float] = {}
    if scalability:
        for method_k, rows in scalability.items():
            method_name = METHOD_MAP.get(method_k, method_k)
            for r in rows:
                lat_lookup[(method_name, int(r["swarm_size"]))] = float(r["avg_time_ms"])

    rows: List[Dict] = []

    for method_k, method_name in METHOD_MAP.items():
        rec = load_json(root / f"method_comparison_{method_k}.json")
        if not rec:
            continue
        seeds = rec.get("seeds", [])
        by_size = rec.get("results_by_size", {})

        for size_s, metrics in by_size.items():
            n = int(size_s)

            acc = metrics.get("task_acceptance", {}).get("raw", [])
            wel = metrics.get("social_welfare", {}).get("raw", [])
            eng = metrics.get("avg_energy", {}).get("raw", [])
            fair = metrics.get("fairness", {}).get("raw", [])

            L = min(len(acc), len(wel), len(eng), len(fair), len(seeds) if seeds else 10**9)
            if L == 10**9:
                L = min(len(acc), len(wel), len(eng), len(fair))

            for i in range(L):
                seed = int(seeds[i]) if i < len(seeds) else i
                rows.append(
                    {
                        "method": method_name,
                        "swarm_size": n,
                        "seed": seed,
                        "acceptance": float(acc[i]),
                        "welfare": float(wel[i]),
                        "energy_kj_per_task": float(eng[i]),
                        "fairness_gini": float(fair[i]),
                        "latency_ms": float(lat_lookup.get((method_name, n), np.nan)),
                    }
                )

    if not rows:
        raise RuntimeError(f"No method comparison files found under {root}")

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Saved {args.output} with {len(out)} rows")


if __name__ == "__main__":
    main()
