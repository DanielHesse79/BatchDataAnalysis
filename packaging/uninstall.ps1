<#
.SYNOPSIS
    Remove shortcuts and, optionally, the virtual environment.

.DESCRIPTION
    Never touches the project source, any database, or generated reports. This
    removes what the installer created, nothing else.

.PARAMETER RemoveEnvironment
    Also delete the .venv directory.
#>

[CmdletBinding()]
param([switch]$RemoveEnvironment)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Batch Insight"
$desktop = [Environment]::GetFolderPath("Desktop")

foreach ($name in @("Batch Insight", "Batch Insight Analyzer", "QC Intelligence Layer")) {
    foreach ($folder in @($startMenu, $desktop)) {
        $path = Join-Path $folder "$name.lnk"
        if (Test-Path $path) { Remove-Item $path -Force; Write-Host "Removed $path" }
    }
}
if ((Test-Path $startMenu) -and -not (Get-ChildItem $startMenu)) { Remove-Item $startMenu -Force }

if ($RemoveEnvironment) {
    $venv = Join-Path $ProjectRoot ".venv"
    if (Test-Path $venv) {
        Remove-Item $venv -Recurse -Force
        Write-Host "Removed $venv"
    }
}

Write-Host "`nDone. Project files, databases and reports were left untouched." -ForegroundColor Green
