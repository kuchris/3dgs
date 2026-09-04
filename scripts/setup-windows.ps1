$ErrorActionPreference = "Stop"

$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not available in PATH. Install uv before running this script."
}
if (-not (Test-Path -LiteralPath $cudaPath)) {
    throw "CUDA Toolkit 13.0 was not found at $cudaPath."
}
if (-not (Test-Path -LiteralPath $vsDevCmd)) {
    throw "Visual Studio 2022 Build Tools were not found at $vsDevCmd."
}

$environmentLines = & $env:ComSpec /s /c "`"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 && set"
foreach ($line in $environmentLines) {
    if ($line -match "^([^=]+)=(.*)$") {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}

$env:DISTUTILS_USE_SDK = "1"
$env:CUDA_HOME = $cudaPath
$env:CUDA_PATH = $cudaPath
$env:PATH = (Join-Path $cudaPath "bin") + [IO.Path]::PathSeparator + $env:PATH
$env:INCLUDE = (Join-Path $cudaPath "include\cccl") + [IO.Path]::PathSeparator + $env:INCLUDE

$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    uv sync --locked
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
