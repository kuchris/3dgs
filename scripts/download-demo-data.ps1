$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$archiveUrl = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
$expectedHash = "d210016bd2de20936a5f02b87fd38a76bf0440c42d045231218372cf9db9a7a1"
$projectRoot = Split-Path $PSScriptRoot -Parent
$destination = Join-Path $projectRoot "data\demo\south-building"
$imageDestination = Join-Path $destination "images"

if (Test-Path -LiteralPath $imageDestination) {
    $existingCount = (Get-ChildItem -LiteralPath $imageDestination -File).Count
    if ($existingCount -eq 128) {
        Write-Output "South Building is already downloaded: $imageDestination"
        exit 0
    }
    throw "The demo dataset is incomplete at $imageDestination"
}
if (Test-Path -LiteralPath $destination) {
    throw "The demo destination already exists but has no images folder: $destination"
}

$temporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) ("capture-studio-data-" + [Guid]::NewGuid())
$archive = Join-Path $temporaryDirectory "south-building.zip"
$unpacked = Join-Path $temporaryDirectory "unpacked"
$stagedDataset = Join-Path $temporaryDirectory "dataset"

New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
try {
    Write-Output "Downloading the 400 MB South Building dataset..."
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archive

    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "The downloaded dataset failed SHA-256 verification."
    }

    Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
    $sourceImages = Join-Path $unpacked "south-building\images"
    $imageCount = (Get-ChildItem -LiteralPath $sourceImages -File).Count
    if ($imageCount -ne 128) {
        throw "Expected 128 South Building photographs but found $imageCount."
    }

    New-Item -ItemType Directory -Path $stagedDataset | Out-Null
    Move-Item -LiteralPath $sourceImages -Destination (Join-Path $stagedDataset "images")
    New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
    Move-Item -LiteralPath $stagedDataset -Destination $destination
    Write-Output "Downloaded 128 photographs to $imageDestination"
}
finally {
    if (Test-Path -LiteralPath $temporaryDirectory) {
        Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force
    }
}
