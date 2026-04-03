Param(
  [string]$OutputDir = "data/derived",
    [string]$MethodResultsDir = "results/new3",
    [string]$Seeds = "0,1,2,3,4",
    [string]$SwarmSizes = "20,50,100,200",
    [int]$SensitivityTasks = 400,
    [int]$RuntimeProfileTasks = 150,
    [switch]$SkipMethodSuite,
    [switch]$SkipSensitivity,
    [switch]$SkipRuntime,
    [switch]$SkipFigureGeneration
)

$ErrorActionPreference = "Stop"

Write-Host "Running only NEW 3 experiments (no full-suite run)."

if (-not (Test-Path $OutputDir)) {
  New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# experiments.py expects --seeds as space-separated ints, not a single comma string
$SeedArgs = @($Seeds -split "[,\s]+" | Where-Object { $_ -and $_.Trim().Length -gt 0 })
if ($SeedArgs.Count -eq 0) {
  $SeedArgs = @("0", "1", "2", "3", "4")
}

# 1) Method suite only (needed for effect-size experiment)
if (-not $SkipMethodSuite) {
    Write-Host "[1/4] Running method comparison only..."
    python "src/experiments.py" `
      --suite method `
      --output_dir $MethodResultsDir `
      --seeds $SeedArgs `
      --device cuda

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Method suite failed."
        exit $LASTEXITCODE
    }

    Write-Host "[1b/4] Converting method JSON -> seed_metrics.csv"
    python "analysis/build_seed_metrics_from_results.py" `
      --results-dir $MethodResultsDir `
      --output "$OutputDir/seed_metrics.csv"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "seed_metrics conversion failed."
        exit $LASTEXITCODE
    }
}

# 2) Sensitivity experiment
if (-not $SkipSensitivity) {
    Write-Host "[2/4] Running sensitivity sweep..."
    python "analysis/run_sensitivity_real_impl.py" `
      --output "$OutputDir/scenario_sensitivity.csv" `
      --seeds $Seeds `
      --arrival-rates "0.5,1.0,1.5,2.0" `
      --deadline-buffers "60,120,180,240,300" `
      --num-drones 200 `
      --tasks $SensitivityTasks `
      --device cuda

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Sensitivity sweep failed."
        exit $LASTEXITCODE
    }
}

# 3) Runtime component breakdown
if (-not $SkipRuntime) {
    Write-Host "[3/4] Running runtime component profiling..."
    python "analysis/runtime_component_profile.py" `
      --output "$OutputDir/runtime_breakdown.csv" `
      --swarm-sizes $SwarmSizes `
      --tasks $RuntimeProfileTasks `
      --device cuda

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Runtime profiling failed."
        exit $LASTEXITCODE
    }
}

# 4) Generate paper placeholder images/tables
if (-not $SkipFigureGeneration) {
    Write-Host "[4/4] Generating placeholder figures from CSV artifacts..."
    python "analysis/generate_additional_figures.py" `
      --seed-metrics "$OutputDir/seed_metrics.csv" `
      --sensitivity "$OutputDir/scenario_sensitivity.csv" `
      --runtime-breakdown "$OutputDir/runtime_breakdown.csv" `
      --output-dir "figures/supplementary"

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Placeholder figure generation failed."
        exit $LASTEXITCODE
    }
}

Write-Host "Done. Generated/updated additional experiment artifacts in: $OutputDir"