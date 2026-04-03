# CUDA Experiment Runner

This folder now includes a full runnable pipeline to generate additional experiment artifacts for the paper.

## Files

- `run_experiments_cuda.py`  
  Runs AL / Q-learning / Greedy / DACA and writes:
  - `seed_metrics.csv`
  - `scenario_sensitivity.csv`
  - `runtime_breakdown.csv`

- `generate_additional_figures.py`  
  Consumes the CSV files and generates paper figures:
  - `effect_size_acceptance.png`
  - `paired_seed_deltas_n200.png`
  - `sensitivity_heatmap_acceptance_daca.png`
  - `runtime_breakdown_components.png`
  - `additional_experiment_summary.csv`

- `run_all_experiments_cuda.ps1`  
  One-shot PowerShell runner (installs deps, runs experiments, generates figures).

## Quick start (Windows PowerShell)

From workspace root:

```powershell
./analysis/run_all_experiments_cuda.ps1
```

By default this writes outputs in the workspace root (`.`), so the generated image names directly match LaTeX placeholders in `paper_body.tex`.

## Lightweight run (only the 3 newly added experiments)

If you do not want to run the full suite, use:

```powershell
./analysis/run_new_3_experiments_only.ps1
```

This runs only:
- method comparison (to build `seed_metrics.csv`),
- sensitivity sweep (`scenario_sensitivity.csv`),
- runtime component profiling (`runtime_breakdown.csv`),
- and then generates placeholder figures/tables.

## CUDA requirement

The runner uses `--require-cuda` and will stop if CUDA is not available.
Install CUDA-enabled PyTorch matching your CUDA version from:
https://pytorch.org/get-started/locally/

## Custom run

```powershell
./analysis/run_all_experiments_cuda.ps1 -OutputDir . -TasksPerRun 2000 -TasksPerSensitivityCell 800 -Seeds "0,1,2,3,4" -SwarmSizes "20,50,100,200,500"
```
