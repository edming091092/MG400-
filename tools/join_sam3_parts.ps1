param(
    [string]$OutputPath = "sam3.pt"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$parts = Get-ChildItem -LiteralPath $root -Filter "sam3.pt.part*" | Sort-Object Name

if (-not $parts) {
    throw "No sam3.pt.part* files found in $root"
}

$out = Join-Path $root $OutputPath
if (Test-Path -LiteralPath $out) {
    Remove-Item -LiteralPath $out -Force
}

$outStream = [System.IO.File]::Open($out, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try {
    foreach ($part in $parts) {
        Write-Host "Appending $($part.Name)"
        $inStream = [System.IO.File]::OpenRead($part.FullName)
        try {
            $inStream.CopyTo($outStream)
        } finally {
            $inStream.Dispose()
        }
    }
} finally {
    $outStream.Dispose()
}

Write-Host "Created $out"
