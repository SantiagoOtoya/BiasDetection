param(
    [int]$Epochs = 4,
    [int]$BatchSize = 16,
    [double]$LearningRate = 2e-5,
    [double]$WarmupRatio = 0.1,
    [double]$ValidationSize = 0.15,
    [double]$OpinionLossAlpha = 0.3,
    [int]$FreezeFirstNLayers = 6,
    [double]$Dropout = 0.2,
    [int]$Seed = 42,
    [string]$Device = "",
    [switch]$Fp16,
    [switch]$InstallRequirements
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ScriptDir

if ($InstallRequirements) {
    python -m pip install -r requirements-finetune.txt
}

$trainArgs = @(
    "finetune_all_mpnet_babe.py",
    "--data-dir", "BABE_HF",
    "--validation-size", "$ValidationSize",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--learning-rate", "$LearningRate",
    "--warmup-ratio", "$WarmupRatio",
    "--opinion-loss-alpha", "$OpinionLossAlpha",
    "--freeze-first-n-layers", "$FreezeFirstNLayers",
    "--dropout", "$Dropout",
    "--output-dir", "models/all-mpnet-base-v2-babe",
    "--monitor-metric", "bias_macro_f1",
    "--early-stopping-patience", "2",
    "--evaluate-test-final",
    "--seed", "$Seed"
)

if ($Device -ne "") {
    $trainArgs += @("--device", $Device)
}

if ($Fp16) {
    $trainArgs += "--fp16"
}

python @trainArgs
