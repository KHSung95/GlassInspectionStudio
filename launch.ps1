<#
.SYNOPSIS  Start the Glass Inspection Studio GUI (optionally opening a project folder).
.EXAMPLE   .\launch.ps1
.EXAMPLE   .\launch.ps1 C:\dev\gis_demo
#>
param([string]$Project = "")
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run .\setup.ps1 first (virtual environment not found)." }
Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_TELEMETRY = "1"
if ($Project) { & $Py -m gistudio $Project } else { & $Py -m gistudio }
