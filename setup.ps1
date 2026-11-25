<#
.SYNOPSIS
  Reproducible setup of Glass Inspection Studio in a project-local virtual environment (.venv).
  The global Python installation is never modified.

.PARAMETER Python          Python launcher selector (default 3.11). Supported: 3.10 - 3.12.
.PARAMETER Torch           auto | cpu | cu118 | cu126 | cu128  (auto = detect NVIDIA driver via nvidia-smi)
.PARAMETER DownloadModels  Download pretrained zero-shot weights (official open_clip release) into .\models
.PARAMETER RunTests        Run the pytest suite after installation
.PARAMETER Recreate        Delete and rebuild .venv (only the venv; no project data is touched)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -DownloadModels -RunTests
#>
param(
    [string]$Python = "3.11",
    [string]$Torch = "auto",
    [switch]$DownloadModels,
    [switch]$RunTests,
    [switch]$Recreate
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$TorchVersion = "2.15.0"
$TorchvisionVersion = "0.29.2"
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

if ($Root.Length -gt 120) {
    Write-Warning "Project path is long ($($Root.Length) chars). PyTorch DLL loading can fail with WinError 206. Use a short path such as C:\dev\GlassInspectionStudio."
}

# ---------------------------------------------------------------- 1. venv
if ($Recreate -and (Test-Path $Venv)) { Remove-Item -Recurse -Force $Venv }
if (-not (Test-Path $Py)) {
    Write-Host "Creating virtual environment with Python $Python ..."
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { & py "-$Python" -m venv $Venv } else { & python -m venv $Venv }
    if (-not (Test-Path $Py)) { throw "venv creation failed (is Python $Python installed?)" }
}
& $Py -c "import sys; v=sys.version_info; assert (3,10)<=v[:2]<=(3,12), v; print('venv python', sys.version)"
& $Py -m pip install --upgrade pip --quiet

# ---------------------------------------------------------------- 2. GPU detection -> torch build
if ($Torch -eq "auto") {
    $Torch = "cpu"
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $hdr = (& nvidia-smi) -join "`n"
        $m = [regex]::Match($hdr, "CUDA Version:\s*([0-9]+)\.([0-9]+)")
        if ($m.Success) {
            $cuda = [double]("{0}.{1}" -f $m.Groups[1].Value, $m.Groups[2].Value)
            Write-Host "NVIDIA driver supports CUDA $cuda"
            if ($cuda -ge 12.8) { $Torch = "cu128" } elseif ($cuda -ge 12.6) { $Torch = "cu126" } elseif ($cuda -ge 11.8) { $Torch = "cu118" }
        }
        & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    } else {
        Write-Host "nvidia-smi not found -> CPU build of PyTorch (no NVIDIA GPU/driver detected)."
    }
}
$Index = "https://download.pytorch.org/whl/$Torch"
Write-Host "Installing torch==$TorchVersion torchvision==$TorchvisionVersion from $Index"
& $Py -m pip install "torch==$TorchVersion" "torchvision==$TorchvisionVersion" --index-url $Index --quiet
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch $TorchVersion is not available for '$Torch'. Edit `$TorchVersion or pass -Torch cpu|cu118|cu126|cu128."
}

# ---------------------------------------------------------------- 3. pinned dependencies + package
& $Py -m pip install -r (Join-Path $Root "requirements.lock.txt") --quiet
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }
& $Py -m pip install -e $Root --no-deps --quiet
& $Py -m gistudio.cli env

# ---------------------------------------------------------------- 4. optional models / tests
if ($DownloadModels) { & $Py (Join-Path $Root "tools\download_models.py") }
if ($RunTests) { & $Py -m pytest -q }
Write-Host "Setup complete. Start the GUI with .\launch.ps1"
