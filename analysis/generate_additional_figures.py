"""
Generate additional experiment figures/tables for the UAV paper.

Expected input files (CSV):
1) seed_metrics.csv
   Required columns:
     method, swarm_size, seed, acceptance, welfare, energy_kj_per_task, fairness_gini, latency_ms

2) scenario_sensitivity.csv (optional but recommended)
   Required columns:
     method, arrival_rate, deadline_buffer_s, seed, acceptance
   Optional columns:
     welfare, energy_kj_per_task

3) runtime_breakdown.csv (optional but recommended)
   Required columns:
     method, swarm_size, bid_compute_ms, auction_resolve_ms, learning_update_ms
   Optional columns:
     comm_overhead_bytes

Outputs (in --output-dir):
- effect_size_acceptance.png
- paired_seed_deltas_n200.png
- sensitivity_heatmap_acceptance_daca.png
- runtime_breakdown_components.png
- additional_experiment_summary.csv

Example:
python analysis/generate_additional_figures.py \
  --seed-metrics results/seed_metrics.csv \
  --sensitivity results/scenario_sensitivity.csv \
  --runtime-breakdown results/runtime_breakdown.csv \
  --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _check_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 3000, alpha: float = 0.05, rng: np.random.Generator | None = None) -> Tuple[float, float, float]:
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        v = float(values[0])
        return v, v, v

    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = values[idx].mean(axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(values.mean()), float(lo), float(hi)


def make_effect_size_plot(seed_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Create bootstrap effect-size figure: DACA - baseline for acceptance."""
    _check_columns(seed_df, ["method", "swarm_size", "seed", "acceptance"], "seed_metrics.csv")

    methods = sorted(seed_df["method"].unique().tolist())
    if "DACA" not in methods:
        raise ValueError("seed_metrics.csv must include method == 'DACA'.")

    baselines = [m for m in methods if m != "DACA"]
    swarm_sizes = sorted(seed_df["swarm_size"].unique().tolist())

    rows = []
    rng = np.random.default_rng(42)

    for n in swarm_sizes:
        daca_n = seed_df[(seed_df["method"] == "DACA") & (seed_df["swarm_size"] == n)]
        for b in baselines:
            base_n = seed_df[(seed_df["method"] == b) & (seed_df["swarm_size"] == n)]
            merged = daca_n[["seed", "acceptance"]].merge(
                base_n[["seed", "acceptance"]], on="seed", suffixes=("_daca", "_base")
            )
            if merged.empty:
                continue
            delta = merged["acceptance_daca"].to_numpy() - merged["acceptance_base"].to_numpy()
            mean, lo, hi = _bootstrap_mean_ci(delta, rng=rng)
            rows.append({"swarm_size": n, "baseline": b, "mean_delta": mean, "ci_low": lo, "ci_high": hi})

    eff = pd.DataFrame(rows)
    if eff.empty:
        raise ValueError("No paired seeds found for DACA vs baselines. Check 'seed' values in seed_metrics.csv.")

    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "^", "D", "P"]
    for i, b in enumerate(sorted(eff["baseline"].unique())):
        sub = eff[eff["baseline"] == b].sort_values("swarm_size")
        yerr = np.vstack([
            sub["mean_delta"].to_numpy() - sub["ci_low"].to_numpy(),
            sub["ci_high"].to_numpy() - sub["mean_delta"].to_numpy(),
        ])
        ax.errorbar(
            sub["swarm_size"],
            sub["mean_delta"],
            yerr=yerr,
            marker=markers[i % len(markers)],
            capsize=4,
            linewidth=2,
            label=f"DACA - {b}",
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Swarm size")
    ax.set_ylabel("Acceptance effect size (percentage points)")
    ax.set_title("Bootstrap effect sizes for acceptance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return eff


def make_paired_seed_delta_plot(seed_df: pd.DataFrame, out_path: Path, swarm_size: int = 200) -> pd.DataFrame:
    _check_columns(seed_df, ["method", "swarm_size", "seed", "acceptance"], "seed_metrics.csv")
    if "DACA" not in seed_df["method"].unique():
        raise ValueError("seed_metrics.csv must include method == 'DACA'.")

    focus = seed_df[seed_df["swarm_size"] == swarm_size].copy()
    baselines = sorted([m for m in focus["method"].unique() if m != "DACA"])
    daca = focus[focus["method"] == "DACA"][["seed", "acceptance"]].rename(columns={"acceptance": "daca"})

    rows = []
    for b in baselines:
        base = focus[focus["method"] == b][["seed", "acceptance"]].rename(columns={"acceptance": "base"})
        m = daca.merge(base, on="seed")
        if m.empty:
            continue
        m["delta"] = m["daca"] - m["base"]
        m["baseline"] = b
        rows.append(m)

    if not rows:
        raise ValueError(f"No paired-seed rows available at swarm_size={swarm_size}.")

    dat = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(dat))
    colors = {b: c for b, c in zip(sorted(dat["baseline"].unique()), ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]) }
    ax.bar(positions, dat["delta"], color=[colors[b] for b in dat["baseline"]])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"seed {s}\nvs {b}" for s, b in zip(dat["seed"], dat["baseline"])], rotation=75, ha="right")
    ax.set_ylabel("Acceptance delta (DACA - baseline, pp)")
    ax.set_title(f"Paired-seed deltas at n={swarm_size}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return dat


def make_sensitivity_heatmap(sens_df: pd.DataFrame, out_path: Path, method: str = "DACA") -> pd.DataFrame:
    _check_columns(sens_df, ["method", "arrival_rate", "deadline_buffer_s", "acceptance"], "scenario_sensitivity.csv")
    sub = sens_df[sens_df["method"] == method].copy()
    if sub.empty:
        raise ValueError(f"scenario_sensitivity.csv has no rows for method == '{method}'.")

    agg = sub.groupby(["arrival_rate", "deadline_buffer_s"], as_index=False)["acceptance"].mean()
    pivot = agg.pivot(index="arrival_rate", columns="deadline_buffer_s", values="acceptance").sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean acceptance (%)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) if float(x).is_integer() else str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("Deadline buffer (s)")
    ax.set_ylabel("Arrival rate λ (tasks/s)")
    ax.set_title(f"Sensitivity heatmap ({method})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return agg


def make_runtime_breakdown_plot(rt_df: pd.DataFrame, out_path: Path, method: str = "DACA") -> pd.DataFrame:
    _check_columns(
        rt_df,
        ["method", "swarm_size", "bid_compute_ms", "auction_resolve_ms", "learning_update_ms"],
        "runtime_breakdown.csv",
    )
    sub = rt_df[rt_df["method"] == method].copy().sort_values("swarm_size")
    if sub.empty:
        raise ValueError(f"runtime_breakdown.csv has no rows for method == '{method}'.")

    x = np.arange(len(sub))
    w = 0.65

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, sub["bid_compute_ms"], width=w, label="Bid compute")
    ax.bar(x, sub["auction_resolve_ms"], width=w, bottom=sub["bid_compute_ms"], label="Auction resolve")
    bottom2 = sub["bid_compute_ms"] + sub["auction_resolve_ms"]
    ax.bar(x, sub["learning_update_ms"], width=w, bottom=bottom2, label="Learning update")

    ax.set_xticks(x)
    ax.set_xticklabels(sub["swarm_size"].astype(str).tolist())
    ax.set_xlabel("Swarm size")
    ax.set_ylabel("Latency per auction (ms)")
    ax.set_title(f"Runtime breakdown by component ({method})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return sub


def build_summary(
    effect_df: pd.DataFrame,
    sens_df: pd.DataFrame | None,
    runtime_df: pd.DataFrame | None,
    out_csv: Path,
) -> None:
    rows: Dict[str, Tuple[float, str]] = {}

    row_200 = effect_df[(effect_df["swarm_size"] == 200)].sort_values("mean_delta", ascending=False)
    if not row_200.empty:
        top = row_200.iloc[0]
        rows["best_effect_size_at_n200_pp"] = (float(top["mean_delta"]), f"DACA - {top['baseline']}")

    if sens_df is not None and not sens_df.empty:
        rows["sensitivity_acceptance_min"] = (float(sens_df["acceptance"].min()), "Across (lambda, buffer) grid")
        rows["sensitivity_acceptance_max"] = (float(sens_df["acceptance"].max()), "Across (lambda, buffer) grid")

    if runtime_df is not None and not runtime_df.empty:
        rmax = runtime_df.sort_values("swarm_size").iloc[-1]
        total = float(rmax["bid_compute_ms"] + rmax["auction_resolve_ms"] + rmax["learning_update_ms"])
        share = float(rmax["learning_update_ms"] / total) if total > 0 else np.nan
        rows["learning_update_share_at_max_n"] = (share, f"n={int(rmax['swarm_size'])}")

    out = pd.DataFrame([
        {"statistic": k, "value": v[0], "notes": v[1]} for k, v in rows.items()
    ])
    out.to_csv(out_csv, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate additional experiment plots for the paper.")
    p.add_argument("--seed-metrics", required=True, help="Path to seed_metrics.csv")
    p.add_argument("--sensitivity", default=None, help="Path to scenario_sensitivity.csv (optional)")
    p.add_argument("--runtime-breakdown", default=None, help="Path to runtime_breakdown.csv (optional)")
    p.add_argument("--output-dir", default="figures", help="Directory for generated figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_df = pd.read_csv(args.seed_metrics)
    effect_df = make_effect_size_plot(seed_df, out_dir / "effect_size_acceptance.png")
    _ = make_paired_seed_delta_plot(seed_df, out_dir / "paired_seed_deltas_n200.png", swarm_size=200)

    sens_agg = None
    if args.sensitivity:
        sens_df = pd.read_csv(args.sensitivity)
        sens_agg = make_sensitivity_heatmap(sens_df, out_dir / "sensitivity_heatmap_acceptance_daca.png", method="DACA")

    runtime_sub = None
    if args.runtime_breakdown:
        rt_df = pd.read_csv(args.runtime_breakdown)
        runtime_sub = make_runtime_breakdown_plot(rt_df, out_dir / "runtime_breakdown_components.png", method="DACA")

    build_summary(
        effect_df=effect_df,
        sens_df=sens_agg,
        runtime_df=runtime_sub,
        out_csv=out_dir / "additional_experiment_summary.csv",
    )

    print(f"Generated outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
