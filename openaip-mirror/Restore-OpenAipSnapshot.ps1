<#
.SYNOPSIS
    Materializes an archived openAIP snapshot back into plain files.

.DESCRIPTION
    Reads a dated snapshot manifest written by Sync-OpenAip.ps1 and expands the
    referenced blobs out of the content-addressed store. This is how you answer
    "what did openAIP say about this airport last March?"

.PARAMETER Root
    Archive root used by Sync-OpenAip.ps1.

.PARAMETER Date
    Snapshot to restore, yyyy-MM-dd. Defaults to the most recent. If no snapshot
    exists for the exact date, the newest one at or before it is used - so you can
    ask for any calendar date without knowing the run schedule.

.PARAMETER Destination
    Where to write the files. Defaults to <Root>\restored\<date>.

.PARAMETER Country / Type / Format
    Optional filters, same vocabulary as Sync-OpenAip.ps1.

.PARAMETER Flat
    Write every file into one directory instead of per-country subfolders.

.EXAMPLE
    .\Restore-OpenAipSnapshot.ps1 -Root D:\Archives\openaip -Date 2026-03-01 -Country us -Format json

.EXAMPLE
    .\Restore-OpenAipSnapshot.ps1 -Root D:\Archives\openaip -List
    Show which snapshots exist.
#>
[CmdletBinding()]
param(
    [string]   $Root        = (Join-Path $PSScriptRoot 'archive'),
    [string]   $Date,
    [string]   $Destination,
    [string[]] $Country     = @(),
    [string[]] $Type        = @(),
    [string[]] $Format      = @(),
    [switch]   $Flat,
    [switch]   $List
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$StoreDir    = Join-Path $Root 'store'
$SnapshotDir = Join-Path $Root 'snapshots'

if (-not (Test-Path -LiteralPath $SnapshotDir)) { throw "no snapshots directory under $Root - has Sync-OpenAip.ps1 run yet?" }

$snaps = @(Get-ChildItem -LiteralPath $SnapshotDir -Filter '*.csv' | Sort-Object Name)
if (-not $snaps.Count) { throw "no snapshots found in $SnapshotDir" }

if ($List) {
    foreach ($s in $snaps) {
        $rows = @(Import-Csv -LiteralPath $s.FullName)
        $m = @($rows) | Measure-Object Size -Sum
        $bytes = if ($null -eq $m -or $null -eq $m.Sum) { [long]0 } else { [long]$m.Sum }
        '{0}  {1,5} files  {2,10:N2} MB (uncompressed)' -f $s.BaseName, $rows.Count, ($bytes / 1MB)
    }
    return
}

# Pick the snapshot: exact match, else the newest at or before the requested date.
if ($Date) {
    $target = $snaps | Where-Object { $_.BaseName -le $Date } | Select-Object -Last 1
    if (-not $target) { throw "no snapshot on or before $Date (earliest is $($snaps[0].BaseName))" }
    if ($target.BaseName -ne $Date) { Write-Host "no snapshot for $Date; using nearest earlier: $($target.BaseName)" -ForegroundColor Yellow }
}
else { $target = $snaps[-1] }

if (-not $Destination) { $Destination = Join-Path (Join-Path $Root 'restored') $target.BaseName }

$rows = @(Import-Csv -LiteralPath $target.FullName)
if ($Country.Count) { $rows = @($rows | Where-Object { $_.Country -in $Country }) }
if ($Type.Count)    { $rows = @($rows | Where-Object { $_.Type    -in $Type }) }
if ($Format.Count)  { $rows = @($rows | Where-Object { $_.Format  -in $Format }) }
if (-not $rows.Count) { throw 'no objects matched those filters' }

Write-Host "restoring $($rows.Count) files from snapshot $($target.BaseName) -> $Destination"

$restored = 0; $missing = 0
foreach ($row in $rows) {
    $blob = Join-Path (Join-Path $StoreDir $row.Sha256.Substring(0, 2)) ($row.Sha256 + '.gz')
    if (-not (Test-Path -LiteralPath $blob)) {
        Write-Warning "blob missing for $($row.Key) ($($row.Sha256)) - was it garbage-collected by -KeepSnapshots?"
        $missing++
        continue
    }

    if ($Flat) { $out = Join-Path $Destination $row.Key }
    else       { $out = Join-Path (Join-Path $Destination $row.Country) $row.Key }

    $dir = Split-Path $out -Parent
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    $in = [IO.File]::OpenRead($blob)
    try {
        $gz = New-Object IO.Compression.GZipStream($in, [IO.Compression.CompressionMode]::Decompress)
        try {
            $fs = [IO.File]::Create($out)
            try { $gz.CopyTo($fs) } finally { $fs.Dispose() }
        }
        finally { $gz.Dispose() }
    }
    finally { $in.Dispose() }

    # The manifest records the byte count the bucket advertised; confirm we rebuilt it.
    $actual = (Get-Item -LiteralPath $out).Length
    if ($actual -ne [long]$row.Size) { Write-Warning "size mismatch on $($row.Key): restored $actual, manifest says $($row.Size)" }
    $restored++
}

Write-Host "restored $restored file(s) to $Destination" -ForegroundColor Green
if ($missing) { Write-Warning "$missing file(s) could not be restored - blobs absent from the store" }
