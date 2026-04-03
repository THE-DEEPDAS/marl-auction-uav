Param(
    [string]$OutputDir = ".",
    [string]$ResultsDir = "results",
    [string]$PlotsDir = "plots",
    [int]$SensitivityTasks = 500,
    [int]$RuntimeProfileTasks = 250,
    [string]$Seeds = "0,1,2,3,4",
    [string]$SwarmSizes = "20,50,100,200,500"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/6] Installing Python dependencies..."
if (Test-Path "requirements.txt") {
    pip install -r "requirements.txt"
}
if (Test-Path "analysis/requirements_additional_experiments.txt") {
    pip install -r "analysis/requirements_additional_experiments.txt"
}

Write-Host "[2/6] Running real experiment implementation (CUDA for DACA where available)..."
python "experiments.py" `
  --suite all `
  --output_dir $ResultsDir `
  --seeds $Seeds `
  --device cuda

if ($LASTEXITCODE -ne 0) {
    Write-Error "Real experiment suite failed."
    exit $LASTEXITCODE
}

Write-Host "[3/6] Generating core plots from real implementation outputs..."
python "analysis.py" `
  --results_dir $ResultsDir `
  --output_dir $PlotsDir

if ($LASTEXITCODE -ne 0) {
    Write-Error "Core plot generation failed."
    exit $LASTEXITCODE
}

Write-Host "[4/6] Building seed_metrics.csv from real outputs..."
python "analysis/build_seed_metrics_from_results.py" `
  --results-dir $ResultsDir `
  --output "$OutputDir/seed_metrics.csv"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build seed_metrics.csv"
    exit $LASTEXITCODE
}

Write-Host "[5/6] Running additional sensitivity + runtime component experiments..."
python "analysis/run_sensitivity_real_impl.py" `
  --output "$OutputDir/scenario_sensitivity.csv" `
  --seeds $Seeds `
  --arrival-rates "0.5,1.0,1.5,2.0" `
  --deadline-buffers "60,120,180,240,300" `
  --num-drones 200 `
  --tasks $SensitivityTasks `
  --device cuda

if ($LASTEXITCODE -ne 0) {
    Write-Error "Sensitivity experiment failed."
    exit $LASTEXITCODE
}

python "analysis/runtime_component_profile.py" `
  --output "$OutputDir/runtime_breakdown.csv" `
  --swarm-sizes $SwarmSizes `
  --tasks $RuntimeProfileTasks `
  --device cuda

if ($LASTEXITCODE -ne 0) {
    Write-Error "Runtime component profiling failed."
    exit $LASTEXITCODE
}

Write-Host "[6/6] Generating paper placeholder figures from CSV artifacts..."
python "analysis/generate_additional_figures.py" `
  --seed-metrics "$OutputDir/seed_metrics.csv" `
  --sensitivity "$OutputDir/scenario_sensitivity.csv" `
  --runtime-breakdown "$OutputDir/runtime_breakdown.csv" `
  --output-dir "$OutputDir"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Placeholder figure generation failed."
    exit $LASTEXITCODE
}

Write-Host "Done. Core results: $ResultsDir ; core plots: $PlotsDir ; additional artifacts: $OutputDir"