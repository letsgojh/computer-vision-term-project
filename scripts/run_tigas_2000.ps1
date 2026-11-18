$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$runOutput = Join-Path (Get-Location) 'outputs/tigas_training_2000'
New-Item -ItemType Directory -Force $runOutput | Out-Null

function Invoke-Stage {
    param([string]$Stage, [string]$Python, [string[]]$Arguments)
    @{ stage = $Stage; state = 'running'; updated_at = (Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content (Join-Path $runOutput 'pipeline_status.json') -Encoding UTF8
    $stageProcess = Start-Process -FilePath $Python -ArgumentList $Arguments -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput (Join-Path $runOutput "$Stage.stdout.log") `
        -RedirectStandardError (Join-Path $runOutput "$Stage.stderr.log")
    if ($stageProcess.ExitCode -ne 0) { throw "$Stage failed with exit code $($stageProcess.ExitCode). See logs." }
}

try {
    Invoke-Stage 'download' '.\.venv-download\Scripts\python.exe' @('-u', '-m', 'src.models.model1_full_image.scripts.download_tigas', '--per-class', '1000')
    Invoke-Stage 'validate' '.\.venv\Scripts\python.exe' @('-u', '-m', 'src.models.model1_full_image.scripts.validate_dataset', '--config', 'configs/model1_tigas_2000.yaml')
    Invoke-Stage 'train' '.\.venv\Scripts\python.exe' @('-u', '-m', 'src.models.model1_full_image.train', '--config', 'configs/model1_tigas_2000.yaml', '--device', 'cuda')
    Invoke-Stage 'evaluate' '.\.venv\Scripts\python.exe' @('-u', '-m', 'src.models.model1_full_image.evaluate', '--config', 'configs/model1_tigas_2000.yaml', '--checkpoint', 'outputs/tigas_training_2000/checkpoints/best_model.pth', '--device', 'cuda')
    @{ stage = 'evaluate'; state = 'completed'; updated_at = (Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content (Join-Path $runOutput 'pipeline_status.json') -Encoding UTF8
} catch {
    @{ state = 'failed'; error = $_.Exception.Message; updated_at = (Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content (Join-Path $runOutput 'pipeline_status.json') -Encoding UTF8
    throw
}
