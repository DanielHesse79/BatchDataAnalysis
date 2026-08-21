<#
.SYNOPSIS
    Set up Batch Insight Analyzer and the QC Intelligence Layer on this machine.

.DESCRIPTION
    Creates a virtual environment, installs dependencies, verifies the install,
    and adds Start Menu and Desktop shortcuts that open each application in its
    own window.

    Everything is per-user. Nothing is written to Program Files, no service is
    registered and no administrator rights are needed, which matters on managed
    laboratory machines where users cannot elevate.

.PARAMETER IncludeOptional
    Also install CatBoost, XGBoost and SHAP. Adds roughly 500 MB.

.PARAMETER IncludeDev
    Also install pytest, so the verification suite can be run.

.PARAMETER NoShortcuts
    Skip creating shortcuts.

.PARAMETER PythonVersion
    Force a specific Python, for example "3.13".

.EXAMPLE
    .\install.ps1
    .\install.ps1 -IncludeOptional -IncludeDev
#>

[CmdletBinding()]
param(
    [switch]$IncludeOptional,
    [switch]$IncludeDev,
    [switch]$NoShortcuts,
    [string]$PythonVersion
)

$ErrorActionPreference = "Stop"

# Versions the project is tested against, best first. 3.14 is deliberately not
# listed: several scientific wheels lag a new release, and a source build on a
# lab machine without a compiler fails in a way that is hard to diagnose.
$SupportedPythonVersions = @("3.13", "3.12", "3.11")

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

function Write-Step { param([string]$Message) Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Good { param([string]$Message) Write-Host "    $Message" -ForegroundColor Green }
function Write-Warn { param([string]$Message) Write-Host "    $Message" -ForegroundColor Yellow }

function Find-Python {
    <# Return the command that starts a supported Python, or $null. #>
    param([string]$Preferred)

    $candidates = if ($Preferred) { @($Preferred) } else { $SupportedPythonVersions }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in $candidates) {
            try {
                $found = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $found) {
                    return [pscustomobject]@{ Command = "py"; Arguments = @("-$version"); Version = $version }
                }
            } catch { }
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $reported = & python -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidates -contains $reported) {
            return [pscustomobject]@{ Command = "python"; Arguments = @(); Version = $reported }
        }
    }

    return $null
}

function New-Shortcut {
    param(
        [string]$Path,
        [string]$TargetPath,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$Description
    )
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($Path)
    $shortcut.TargetPath = $TargetPath
    $shortcut.Arguments = $Arguments
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = $Description
    $shortcut.Save()
}

Write-Host "Batch Insight - installation" -ForegroundColor White
Write-Host "Project: $ProjectRoot"

# ---------------------------------------------------------------- 1. Python
Write-Step "Locating a supported Python"
$python = Find-Python -Preferred $PythonVersion

if (-not $python) {
    Write-Host @"

    No supported Python was found.

    This project needs Python $($SupportedPythonVersions -join ", ").
    Install one from https://www.python.org/downloads/ or the Microsoft Store,
    tick "Add python.exe to PATH", then run this installer again.

    If Python is installed but not detected, pass it explicitly:
        .\install.ps1 -PythonVersion 3.13
"@ -ForegroundColor Red
    exit 1
}

$pythonDisplay = (& $python.Command @($python.Arguments) -c "import sys; print(sys.version.split()[0])").Trim()
Write-Good "Python $pythonDisplay"

# ------------------------------------------------------------------ 2. venv
Write-Step "Preparing the virtual environment"
if (Test-Path $VenvPython) {
    $existing = (& $VenvPython -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))").Trim()
    Write-Good "Reusing the existing environment (Python $existing)"
} else {
    & $python.Command @($python.Arguments) -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment at $VenvPath" }
    Write-Good "Created $VenvPath"
}

# ---------------------------------------------------------- 3. dependencies
Write-Step "Installing dependencies (this takes a few minutes on first run)"
& $VenvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip" }

$requirementFiles = @("requirements.txt", "requirements-desktop.txt")
if ($IncludeDev) { $requirementFiles += "requirements-dev.txt" }
if ($IncludeOptional) { $requirementFiles += "requirements-optional.txt" }

foreach ($file in $requirementFiles) {
    $path = Join-Path $ProjectRoot $file
    if (-not (Test-Path $path)) { Write-Warn "Skipping $file (not present)"; continue }

    Write-Host "    installing $file ..."
    & $VenvPython -m pip install -r $path --quiet --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "Failed while installing $file" }
    Write-Good "$file"
}

# ----------------------------------------------------------- 4. verification
Write-Step "Verifying the installation"
# importlib.util must be imported explicitly; importing importlib alone does not
# bind the submodule, and the resulting AttributeError looks like a broken install.
$verifyScript = @"
import importlib.util, sys
required = ['streamlit', 'pandas', 'numpy', 'sklearn', 'scipy', 'plotly', 'reportlab']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print('MISSING: ' + ', '.join(missing)); sys.exit(1)
import qc_intel, analysis, ui
print('OK')
"@
$verifyResult = & $VenvPython -c $verifyScript 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Verification failed: $verifyResult" -ForegroundColor Red
    exit 1
}
Write-Good "All required packages import cleanly"

& $VenvPython -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('webview') else 1)" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Good "Desktop window support available"
} else {
    Write-Warn "pywebview not installed - the apps will open in your browser instead"
}

# ------------------------------------------------------------- 5. shortcuts
if (-not $NoShortcuts) {
    Write-Step "Creating shortcuts"
    $pythonwExe = Join-Path $VenvPath "Scripts\pythonw.exe"
    $launcher = Join-Path $ProjectRoot "launcher.py"
    $startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Batch Insight"
    New-Item -ItemType Directory -Force -Path $startMenu | Out-Null

    $shortcuts = @(
        @{ Name = "Batch Insight";          App = "home";  Description = "Choose between batch driver analysis and QC monitoring" },
        @{ Name = "Batch Insight Analyzer"; App = "batch"; Description = "Link batch process parameters to QC outcomes" },
        @{ Name = "QC Intelligence Layer";  App = "qc";    Description = "Trend QC drift across runs, instruments and methods" }
    )

    foreach ($entry in $shortcuts) {
        foreach ($folder in @($startMenu, [Environment]::GetFolderPath("Desktop"))) {
            New-Shortcut `
                -Path (Join-Path $folder "$($entry.Name).lnk") `
                -TargetPath $pythonwExe `
                -Arguments "`"$launcher`" --app $($entry.App)" `
                -WorkingDirectory $ProjectRoot `
                -Description $entry.Description
        }
        Write-Good $entry.Name
    }
}

# ------------------------------------------------------------------- done
Write-Host @"

Installation complete.

  Start from the Start Menu or Desktop shortcut, or run:
      .\.venv\Scripts\python.exe launcher.py --app batch
      .\.venv\Scripts\python.exe launcher.py --app qc

  Generate the QC prototype's synthetic data first if you want to explore it:
      .\.venv\Scripts\python.exe -m qc_intel.build_prototype

"@ -ForegroundColor Green

if ($IncludeDev) {
    Write-Host "  Run the verification suite:" -ForegroundColor Green
    Write-Host "      .\.venv\Scripts\python.exe -m pytest -q`n" -ForegroundColor Green
}
