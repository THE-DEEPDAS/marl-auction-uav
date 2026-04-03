# Drone_Paper (Submission Layout)

This repository is organized for clean reproducibility and paper submission.

## Structure

- `main.tex`, `paper_body.tex` — manuscript source.
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

Use existing archived outputs whenever possible.

- Generate supplementary figures from CSVs:
  - `python analysis/generate_additional_figures.py --seed-metrics data/derived/seed_metrics.csv --sensitivity data/derived/scenario_sensitivity.csv --runtime-breakdown data/derived/runtime_breakdown.csv --output-dir figures/supplementary`

- Run only the three additional experiments:
  - `./analysis/run_new_3_experiments_only.ps1`

## Code and project link

https://github.com/THE-DEEPDAS/marl-auction-uav
