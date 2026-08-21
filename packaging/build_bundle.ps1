<#
.SYNOPSIS
    Build a self-contained Batch Insight bundle that needs no Python.

.DESCRIPTION
    Produces dist\BatchInsight\, a folder containing BatchInsight.exe and
    everything it needs. A user copies the folder and double-clicks the
    executable. Nothing is installed and no interpreter is required.

    This is the build step, run by a developer on a machine that has Python and
    the project's dependencies. Users never run it.

.PARAMETER Zip
    Also produce dist\BatchInsight-<version>.zip for distribution.

.PARAMETER Clean
    Delete previous build and dist output first.

.EXAMPLE
    .\build_bundle.ps1 -Clean -Zip
#>

[CmdletBinding()]
param(
    [switch]$Zip,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SpecFile = Join-Path $PSScriptRoot "batch_insight.spec"
$DistDir = Join-Path $ProjectRoot "dist"
$BuildDir = Join-Path $ProjectRoot "build"
$BundleDir = Join-Path $DistDir "BatchInsight"

function Write-Step { param([string]$Message) Write-Host "`n==> $Message" -ForegroundColor Cyan }

if (-not (Test-Path $VenvPython)) {
    Write-Host "No virtual environment found. Run packaging\install.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Step "Checking the build toolchain"
& $VenvPython -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    installing PyInstaller ..."
    & $VenvPython -m pip install pyinstaller --quiet --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyInstaller" }
}
$pyinstallerVersion = (& $VenvPython -c "import PyInstaller; print(PyInstaller.__version__)").Trim()
Write-Host "    PyInstaller $pyinstallerVersion" -ForegroundColor Green

# The example dataset is generated, not committed, so a clean checkout has none.
# Without it an installed copy opens on an error instead of a demonstration.
$ExampleDatabase = Join-Path $ProjectRoot "qc_intel\data\qc_intel.sqlite"
if (-not (Test-Path $ExampleDatabase)) {
    Write-Step "Generating the example QC dataset"
    & $VenvPython -m qc_intel.synth.generate
    if ($LASTEXITCODE -ne 0) { throw "Could not generate the example dataset" }
    & $VenvPython -m qc_intel.build_prototype
    if ($LASTEXITCODE -ne 0) { throw "Could not build the example database" }
}

if ($Clean) {
    Write-Step "Cleaning previous output"
    foreach ($path in @($BuildDir, $DistDir)) {
        if (Test-Path $path) { Remove-Item $path -Recurse -Force; Write-Host "    removed $path" }
    }
}

Write-Step "Building (several minutes; the scientific stack is large)"
& $VenvPython -m PyInstaller $SpecFile `
    --distpath $DistDir --workpath $BuildDir --noconfirm --log-level WARN
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

if (-not (Test-Path (Join-Path $BundleDir "BatchInsight.exe"))) {
    throw "The build finished but BatchInsight.exe is missing from $BundleDir"
}

$sizeMb = (Get-ChildItem $BundleDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("    bundle size: {0:N0} MB" -f $sizeMb) -ForegroundColor Green

Write-Step "Checking the bundle for missing dependencies"
# A frozen build only contains what PyInstaller found by following imports. The
# application packages ship as files, so a dependency reached only from app code
# can be absent and stay absent until a user opens the page that needs it.
$ImportLog = Join-Path $BuildDir "import-check.txt"
$check = Start-Process -FilePath (Join-Path $BundleDir "BatchInsight.exe") `
    -ArgumentList "--check-imports" -Wait -PassThru -RedirectStandardError $ImportLog
if ($check.ExitCode -ne 0) {
    Write-Host "    modules the bundle cannot import:" -ForegroundColor Red
    Get-Content $ImportLog | ForEach-Object { Write-Host "      $_" -ForegroundColor Red }
    throw "The bundle is incomplete. Add the missing packages to batch_insight.spec."
}
Write-Host "    every shipped module imports" -ForegroundColor Green

Write-Step "Adding the launcher shortcuts"
# A shared chooser plus two direct-entry wrappers for frequent users.
@(
    @{ File = "Open Batch Insight.cmd";      App = "home" },
    @{ File = "Batch Insight Analyzer.cmd"; App = "batch" },
    @{ File = "QC Intelligence Layer.cmd";  App = "qc" }
) | ForEach-Object {
    $content = "@echo off`r`nstart """" ""%~dp0BatchInsight.exe"" --app $($_.App)`r`n"
    Set-Content -Path (Join-Path $BundleDir $_.File) -Value $content -Encoding ASCII
    Write-Host "    $($_.File)" -ForegroundColor Green
}

Set-Content -Path (Join-Path $BundleDir "READ ME FIRST.txt") -Encoding UTF8 -Value @"
Batch Insight
=============

Double-click Open Batch Insight.cmd to choose a workspace, or use a direct shortcut:

    Open Batch Insight.cmd          Choose between both analysis workspaces
    Batch Insight Analyzer.cmd     Link batch process parameters to QC outcomes
    QC Intelligence Layer.cmd      Trend QC drift across runs and instruments

Nothing needs to be installed. No Python, no dependencies, no administrator
rights. The first start takes a few seconds while the application unpacks.

If Windows shows "Windows protected your PC", choose More info and then Run
anyway. That warning appears because this build is not code-signed, not
because anything is wrong with it.

These applications are decision-support tools. They are not validated systems
and must not be used for run acceptance or regulatory reporting.
"@

if ($Zip) {
    Write-Step "Creating the distribution archive"
    $version = (& $VenvPython -c "import sys; sys.path.insert(0,r'$ProjectRoot'); import qc_intel; print(qc_intel.__version__)").Trim()
    $zipPath = Join-Path $DistDir "BatchInsight-$version.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path $BundleDir -DestinationPath $zipPath -CompressionLevel Optimal
    $zipMb = (Get-Item $zipPath).Length / 1MB
    Write-Host ("    {0} ({1:N0} MB)" -f $zipPath, $zipMb) -ForegroundColor Green
}

Write-Host @"

Build complete.

  Bundle:  $BundleDir
  Test it: & "$BundleDir\BatchInsight.exe" --app qc

  Distribute the folder, or the zip if you built one. Users double-click
  "Batch Insight Analyzer.cmd" or "QC Intelligence Layer.cmd".

"@ -ForegroundColor Green
