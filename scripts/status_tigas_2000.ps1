param([switch]$Watch)

$ErrorActionPreference = 'Stop'
$runOutput = Join-Path (Split-Path $PSScriptRoot -Parent) 'outputs/tigas_training_2000'

do {
    $status = Get-Content (Join-Path $runOutput 'pipeline_status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $progress = [ordered]@{
        stage = $status.stage
        state = $status.state
    }
    if ($status.error) { $progress.error = $status.error }
    if ($status.stage -eq 'download') {
        $logPath = Join-Path $runOutput 'download.stderr.log'
        $recentLog = (Get-Content $logPath -Tail 20 -Encoding UTF8) -join "`n"
        $entries = [regex]::Matches($recentLog, '(\d+)/(\d+)\s+\[([^\]]+)\]')
        if ($entries.Count -gt 0) {
            $latest = $entries[$entries.Count - 1]
            $completed = [int]$latest.Groups[1].Value
            $total = [int]$latest.Groups[2].Value
            $progress.completed_images = $completed
            $progress.total_images = $total
            $progress.download_percent = [math]::Round(100.0 * $completed / $total, 1)
            $progress.timing = $latest.Groups[3].Value
        }
    } elseif ($status.stage -eq 'train') {
        $logPath = Join-Path $runOutput 'train.stdout.log'
        $recentLog = (Get-Content $logPath -Tail 20 -Encoding UTF8) -join "`n"
        $entries = [regex]::Matches($recentLog, 'Epoch (\d+)/(\d+):')
        if ($entries.Count -gt 0) {
            $latest = $entries[$entries.Count - 1]
            $progress.completed_epochs = [int]$latest.Groups[1].Value
            $progress.max_epochs = [int]$latest.Groups[2].Value
            $progress.epoch_percent = [math]::Round(100.0 * $progress.completed_epochs / $progress.max_epochs, 1)
        }
    }
    $progress.checked_at = (Get-Date).ToString('o')
    if ($Watch) { Clear-Host }
    $progress | ConvertTo-Json
    if ($Watch -and $status.state -eq 'running') { Start-Sleep -Seconds 3 }
} while ($Watch -and $status.state -eq 'running')
