$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$version = "4.2.0"
$expectedHash = "991e0bae403a496fcc4de0c1f1f428619bf12f8000978f77bc6799d9bfeac23e"
$archiveUrl = "https://github.com/colmap/colmap/releases/download/$version/colmap-x64-windows-cuda.zip"
$installDirectory = Join-Path $env:LOCALAPPDATA "Programs\COLMAP-$version"
$executable = Join-Path $installDirectory "COLMAP.bat"

if (Test-Path -LiteralPath $executable) {
    Write-Output "COLMAP $version is already installed at $installDirectory"
    exit 0
}
if (Test-Path -LiteralPath $installDirectory) {
    throw "The install directory exists but COLMAP.bat is missing: $installDirectory"
}

$temporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) ("capture-studio-colmap-" + [Guid]::NewGuid())
$archive = Join-Path $temporaryDirectory "colmap.zip"
$unpackedDirectory = Join-Path $temporaryDirectory "unpacked"

New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
try {
    Write-Output "Downloading COLMAP $version..."
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archive

    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "The downloaded COLMAP archive failed SHA-256 verification."
    }

    Expand-Archive -LiteralPath $archive -DestinationPath $unpackedDirectory
    if (-not (Test-Path -LiteralPath (Join-Path $unpackedDirectory "COLMAP.bat"))) {
        throw "The COLMAP archive has an unexpected layout."
    }

    $programsDirectory = Split-Path $installDirectory -Parent
    New-Item -ItemType Directory -Path $programsDirectory -Force | Out-Null
    Move-Item -LiteralPath $unpackedDirectory -Destination $installDirectory
    Write-Output "Installed COLMAP $version at $installDirectory"
}
finally {
    if (Test-Path -LiteralPath $temporaryDirectory) {
        Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force
    }
}
