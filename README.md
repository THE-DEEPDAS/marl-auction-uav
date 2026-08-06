# Drone_Paper (Submission Layout)

This repository is organized for clean reproducibility and paper submission.

## Structure

- `main.tex`, `final_body.tex` — active manuscript source. `paper_body.tex` is an archived pre-correction draft and is not included by `main.tex`.
- `src/` — core implementation:
  - `agents.py`
  - `simulator.py`
  - `experiments.py`
  - `analysis.py`
- `analysis/` — experiment/plot orchestration scripts.
- `figures/main/` — figures used in the main paper.
- `figures/supplementary/` — additional/supplementary figures.
- `data/derived/` — generated CSV artifacts used by supplementary analyses.
- `results/` — archived canonical and auxiliary experiment outputs:
  - `results/all_results.json`, `results/scalability.json`, etc. (cited outputs)
  - `results/new3/` (additional-experiment method suite outputs)
  - `results/smoke/` and `results/smoke/plots/` (smoke-test outputs)

## Reproduce key supplementary artifacts

## Corrected evaluation protocol

The current manuscript uses the calibrated capability-aware DACA protocol in `results/typeaware_travel02_60s_10seed_cuda/` and `results/final_stress_cuda/`: the evaluated bid is a local utility prior using each UAV's observed speed and energy type, with no simulator valuation target and no uncalibrated exploration noise. Updates use post-auction observations as next states, and task destinations are applied when queued completion events occur. The nominal validation uses ten seeds at 100 UAVs; the workload stress matrix uses five seeds across sparse, nominal, and loaded streams. Runs use the PowerMind CUDA environment (`powermind_rtx5050`, `device='cuda'`). The archived pre-correction outputs must not be used for claims about learning or convergence.

The active LaTeX source is `paper/main.tex` with `paper/final_body.tex`; the verified nominal comparison figure is `paper/nominal_objective_comparison.png`.

Use existing archived outputs whenever possible.

- Generate supplementary figures from CSVs:
  - `python analysis/generate_additional_figures.py --seed-metrics data/derived/seed_metrics.csv --sensitivity data/derived/scenario_sensitivity.csv --runtime-breakdown data/derived/runtime_breakdown.csv --output-dir figures/supplementary`

- Run only the three additional experiments:
  - `./analysis/run_new_3_experiments_only.ps1`

## Code and project link

https://github.com/THE-DEEPDAS/marl-auction-uav
