param(
    [string]$Version = '23.4'
)

$ErrorActionPreference = 'Stop'

Write-Host "Installing protoc version $Version into src-tauri/.protoc..."

$outDir = Join-Path -Path $PSScriptRoot -ChildPath "..\src-tauri\.protoc"
if (-not (Test-Path -Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}
$outDir = Resolve-Path -Path $outDir | Select-Object -ExpandProperty Path

$zipName = "protoc-$Version-win64.zip"
$url = "https://github.com/protocolbuffers/protobuf/releases/download/v$Version/$zipName"
$zipPath = Join-Path -Path $env:TEMP -ChildPath $zipName

Write-Host "Downloading $url..."
try {
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing -ErrorAction Stop
} catch {
    Write-Error "Failed to download protoc from $url. Check version or network. $_"
    exit 1
}

Write-Host "Extracting to $outDir..."
try {
    if (Test-Path -Path $outDir) {
        Remove-Item -Path $outDir -Recurse -Force
        New-Item -ItemType Directory -Path $outDir | Out-Null
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $outDir)
} catch {
    Write-Error ("Failed to extract {0}: {1}" -f $zipPath, $_)
    exit 1
}

Remove-Item -Path $zipPath -Force

$protocPath = Join-Path -Path $outDir -ChildPath "bin\protoc.exe"
if (Test-Path -Path $protocPath) {
    Write-Host "protoc installed at: $protocPath"
    Write-Host "You can now run `pnpm tauri dev`. Cargo is configured to use this protoc via .cargo/config.toml."
    exit 0
} else {
    Write-Error "protoc.exe not found after extraction. Directory contents:" 
    Get-ChildItem -Path $outDir -Recurse
    exit 1
}
