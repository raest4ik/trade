param(
    [string]$RepoRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

Push-Location $RepoRoot
try {
    docker compose up -d postgres | Out-Null
    $arguments = @("run", "python", "-m", "apps.cli.live_corpus_run", "--repo-root", $RepoRoot)
    if ($DryRun) {
        $arguments += "--dry-run"
    }
    & uv @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
