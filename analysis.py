"""Plotting and summary generation for experiment outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 11
mpl.rcParams["font.family"] = "serif"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COLOR_MAP = {
    "daca": "#006d77",
    "qlearning": "#e76f51",
    "auction_nolearning": "#264653",
    "greedy": "#8d99ae",
}

METHOD_NAMES = {
    "daca": "DACA (Ours)",
    "qlearning": "MARL-B (Q-learning)",
    "auction_nolearning": "AL (No Learning)",
    "greedy": "Greedy",
}

ORDERED_METHODS = ["auction_nolearning", "qlearning", "greedy", "daca"]
SWARM_SIZES = [20, 50, 100, 200]


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(results_dir: str = "results") -> Dict[str, Dict]:
    root = Path(results_dir)
    out: Dict[str, Dict] = {}

    out["convergence"] = _load_json(root / "convergence.json")
    out["real_convergence"] = _load_json(root / "real_convergence.json")
    out["scalability"] = _load_json(root / "scalability.json")
    out["ablation"] = _load_json(root / "ablation.json")
    out["all"] = _load_json(root / "all_results.json")

    for m in ORDERED_METHODS:
        out[f"method_comparison_{m}"] = _load_json(root / f"method_comparison_{m}.json")

    return out


def _series(all_results: Dict, method: str, metric: str) -> List[float]:
    results = all_results.get(f"method_comparison_{method}", {})
    by_size = results.get("results_by_size", {})
    vals = []
    for size in SWARM_SIZES:
        rec = by_size.get(str(size), {})
        vals.append(float(rec.get(metric, {}).get("mean", 0.0)))
    return vals


def plot_metric(all_results: Dict, metric: str, ylabel: str, title: str, output_file: str, ylog: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ORDERED_METHODS:
        vals = _series(all_results, method, metric)
        ax.plot(
            SWARM_SIZES,
            vals,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=COLOR_MAP[method],
            label=METHOD_NAMES[method],
        )
    ax.set_xlabel("Swarm Size (# Drones)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if ylog:
        ax.set_yscale("log")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_file)


def plot_convergence(all_results: Dict, output_file: str) -> None:
    conv = all_results.get("convergence", {})
    if not conv:
        logger.warning("Convergence file not found, skipping convergence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ORDERED_METHODS:
        rec = conv.get(method, {})
        x = rec.get("num_tasks", [])
        y = rec.get("convergence_distances", [])
        if x and y:
            ax.loglog(x, y, marker="o", linewidth=2.4, markersize=7, color=COLOR_MAP[method], label=METHOD_NAMES[method])

    x_ref = np.array([100, 300, 1000, 3000, 10000], dtype=np.float64)
    y_ref = np.log(x_ref) / np.sqrt(x_ref)
    y_ref = y_ref / y_ref[0]
    ax.loglog(x_ref, y_ref, "k--", linewidth=2.0, label=r"Theory: $O(\log t/\sqrt{t})$")

    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel(r"Equilibrium Distance $\Psi(t)$")
    ax.set_title("Convergence to Approximate Nash Equilibrium")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_file)


def plot_real_convergence(all_results: Dict, output_file: str) -> None:
    data = all_results.get("real_convergence", {})
    methods = data.get("methods", {}) if isinstance(data, dict) else {}
    if not methods:
        logger.warning("Real convergence file not found, skipping real convergence plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    metrics = [
        ("lyapunov_mean", "lyapunov_std", r"$\Psi(t)$ (Bid-Truth MSE)"),
        ("best_response_gain_mean", "best_response_gain_std", "Best-Response Gain"),
        ("bid_mae_mean", "bid_mae_std", "Bid Truthfulness MAE"),
        ("efficiency_gap_mean", "efficiency_gap_std", "Allocation Efficiency Gap"),
    ]

    for ax, (m_key, s_key, ylab) in zip(axes, metrics):
        for method in ORDERED_METHODS:
            rec = methods.get(method, {})
            x = rec.get("window_midpoints", [])
            y = rec.get(m_key, [])
            s = rec.get(s_key, [])
            if not x or not y:
                continue
            xx = np.array(x, dtype=np.float64)
            yy = np.array(y, dtype=np.float64)
            ss = np.array(s, dtype=np.float64) if s else np.zeros_like(yy)

            ax.plot(xx, yy, linewidth=2.1, color=COLOR_MAP[method], label=METHOD_NAMES[method])
            ax.fill_between(xx, np.maximum(yy - ss, 0.0), yy + ss, alpha=0.16, color=COLOR_MAP[method])

        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)

    axes[2].set_xlabel("Tasks Processed")
    axes[3].set_xlabel("Tasks Processed")
    axes[0].set_title("Real Convergence Diagnostics")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_file)


def plot_scalability(all_results: Dict, output_file: str) -> None:
    data = all_results.get("scalability", {})
    if not data:
        logger.warning("Scalability file not found, skipping scalability plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ORDERED_METHODS:
        rows = data.get(method, [])
        if not rows:
            continue
        x = [int(r["swarm_size"]) for r in rows]
        y = [float(r["avg_time_ms"]) for r in rows]
        ax.plot(x, y, marker="D", linewidth=2.4, markersize=7, color=COLOR_MAP[method], label=METHOD_NAMES[method])

    ax.set_xlabel("Swarm Size (# Drones)")
    ax.set_ylabel("Wall-Clock Time per Auction (ms)")
    ax.set_title("Computational Scalability")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_file)


def plot_ablation(all_results: Dict, output_file: str) -> None:
    ab = all_results.get("ablation", {})
    if not ab:
        logger.warning("Ablation file not found, skipping ablation plot")
        return

    rec = ab.get("results", {})
    if not rec:
        logger.warning("Ablation results empty, skipping ablation plot")
        return

    labels = list(rec.keys())
    vals = [float(rec[k]) for k in labels]
    colors = ["#006d77" if i == 0 else "#adb5bd" for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2.0, f"{val:.2f}%", va="center")
    ax.set_xlabel("Task Acceptance Rate (%)")
    ax.set_title("Ablation Study (n=100)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_file)


def save_summary_tables(all_results: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    lines = []

    lines.append("# Experimental Summary\n")
    lines.append("## Task Acceptance Rate (%)\n")
    lines.append("| Method | 20 | 50 | 100 | 200 |")
    lines.append("|---|---:|---:|---:|---:|")
    for method in ORDERED_METHODS:
        vals = _series(all_results, method, "task_acceptance")
        lines.append(f"| {METHOD_NAMES[method]} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {vals[3]:.2f} |")

    lines.append("\n## Average Energy (kJ/task)\n")
    lines.append("| Method | 20 | 50 | 100 | 200 |")
    lines.append("|---|---:|---:|---:|---:|")
    for method in ORDERED_METHODS:
        vals = _series(all_results, method, "avg_energy")
        lines.append(f"| {METHOD_NAMES[method]} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} |")

    lines.append("\n## Social Welfare\n")
    lines.append("| Method | 20 | 50 | 100 | 200 |")
    lines.append("|---|---:|---:|---:|---:|")
    for method in ORDERED_METHODS:
        vals = _series(all_results, method, "social_welfare")
        lines.append(f"| {METHOD_NAMES[method]} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} |")

    out_path = Path(output_dir) / "summary_tables.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved %s", out_path)


def generate_all_plots(results_dir: str = "results", output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    all_results = load_results(results_dir)

    plot_metric(
        all_results,
        metric="task_acceptance",
        ylabel="Task Acceptance Rate (%)",
        title="Task Acceptance vs Swarm Size",
        output_file=os.path.join(output_dir, "task_acceptance.png"),
    )
    plot_metric(
        all_results,
        metric="avg_energy",
        ylabel="Average Energy (kJ/task)",
        title="Energy Efficiency vs Swarm Size",
        output_file=os.path.join(output_dir, "energy_consumption.png"),
    )
    plot_metric(
        all_results,
        metric="social_welfare",
        ylabel="Normalized Social Welfare",
        title="Social Welfare vs Swarm Size",
        output_file=os.path.join(output_dir, "social_welfare.png"),
    )
    plot_metric(
        all_results,
        metric="fairness",
        ylabel="Fairness Index (Gini, lower better)",
        title="Fairness vs Swarm Size",
        output_file=os.path.join(output_dir, "fairness.png"),
    )
    plot_convergence(all_results, os.path.join(output_dir, "convergence_rates.png"))
    plot_real_convergence(all_results, os.path.join(output_dir, "real_convergence_diagnostics.png"))
    plot_scalability(all_results, os.path.join(output_dir, "scalability.png"))
    plot_ablation(all_results, os.path.join(output_dir, "ablation_study.png"))
    save_summary_tables(all_results, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures and summary tables")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
