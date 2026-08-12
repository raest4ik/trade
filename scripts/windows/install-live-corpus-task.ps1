param(
    [string]$TaskName = "Trade AI Live Corpus",
    [ValidateRange(15, 1440)]
    [int]$IntervalMinutes = 60,
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$runner = Join-Path $RepoRoot "scripts\windows\run-live-corpus.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Runner script not found: $runner"
}

$escapedRunner = $runner.Replace('"', '`"')
$escapedRoot = $RepoRoot.Replace('"', '`"')
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$escapedRunner`" -RepoRoot `"$escapedRoot`"" `
    -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At ((Get-Date).AddMinutes(1)) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Hourly zero-cost Trade AI REAL corpus collection; never trains a model." `
    -Force

Write-Output "Installed '$TaskName' every $IntervalMinutes minutes for current interactive user."
