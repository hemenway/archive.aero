<#
.SYNOPSIS
    Mirrors the openAIP system-export bucket into a de-duplicated, verifiable local archive.

.DESCRIPTION
    Enumerates every object in the public openAIP export bucket (S3 ListObjectsV2),
    downloads whatever changed since the last run, and records a dated snapshot manifest.

    Storage model - content-addressed, so daily runs are cheap:

        <Root>\store\<ab>\<sha256>.gz     one gzipped copy per UNIQUE file content
        <Root>\snapshots\<yyyy-MM-dd>.csv  what the bucket looked like on that date
        <Root>\logs\<yyyy-MM-dd>.log       run log

    Most country files change rarely, so an unchanged file costs zero extra bytes on
    day 2 - the new snapshot just points at the blob already in the store. Restoring
    any past day is a manifest lookup (see Restore-OpenAipSnapshot.ps1).

    INTEGRITY: this bucket is observed to silently truncate large transfers - the
    server returns HTTP 200 and simply stops sending bytes mid-file. Every download is
    therefore checked against Content-Length and resumed via HTTP Range until complete;
    a short file is never written into the store.

.PARAMETER Root
    Archive directory. Holds store\, snapshots\ and logs\. The whole bucket is about
    1.6 GB uncompressed (5,275 objects, measured 2026-09-17); gzipped in the store it
    lands near 300-400 MB, and later snapshots only add what actually changed.

.PARAMETER Country
    Two-letter country codes to include, e.g. -Country us,ca,mx. Default: all.

.PARAMETER Type
    Object types to include: apt, asp, nav, obs, rpp, hgl, hot, rca, raa (and the
    versioned asp_v1/asp_v2/raa_v1/raa_v2 OpenAIR variants). Default: all.

.PARAMETER Format
    File extensions to include: json, geojson, ndgeojson, cup, cupx, xml, txt.
    Default: all.

.PARAMETER ListOnly
    Enumerate and report totals (object count, bytes) without downloading anything.

.PARAMETER Materialize
    Also write a plain, uncompressed copy of the current state to <Root>\current\,
    laid out as <country>\<filename>. Convenient if other tools read the files
    directly; roughly doubles disk usage.

.PARAMETER KeepSnapshots
    Keep only the N most recent snapshots and garbage-collect blobs no surviving
    snapshot references. 0 (default) keeps everything forever.

.EXAMPLE
    .\Sync-OpenAip.ps1 -Root D:\Archives\openaip -ListOnly
    Size the job before committing disk.

.EXAMPLE
    .\Sync-OpenAip.ps1 -Root D:\Archives\openaip -Country us,ca -Format json,geojson
    Mirror just North American JSON/GeoJSON.

.EXAMPLE
    .\Sync-OpenAip.ps1 -Root D:\Archives\openaip
    Full mirror. Safe to run daily; only changed files transfer.
#>
[CmdletBinding()]
param(
    [string]   $Root          = (Join-Path $PSScriptRoot 'archive'),
    [string[]] $Country       = @(),
    [string[]] $Type          = @(),
    [string[]] $Format        = @(),
    [switch]   $ListOnly,
    [switch]   $Materialize,
    [int]      $KeepSnapshots = 0,
    [int]      $MaxRetries    = 6,
    [string]   $BaseUrl       = 'https://storage.openaip.net/openaip-system-exports'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
# The bucket is fronted by envoy; more than a few parallel streams draws throttling.
[Net.ServicePointManager]::DefaultConnectionLimit = 8

$script:UserAgent = 'openaip-archive-mirror/1.0 (+personal archival)'
$script:LogFile   = $null

#region helpers -------------------------------------------------------------

function Write-Log {
    param([string] $Message, [string] $Level = 'INFO')
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    switch ($Level) {
        'WARN'  { Write-Host $line -ForegroundColor Yellow }
        'ERROR' { Write-Host $line -ForegroundColor Red }
        default { Write-Host $line }
    }
    if ($script:LogFile) { Add-Content -LiteralPath $script:LogFile -Value $line -Encoding utf8 }
}

function Format-Bytes {
    param([long] $Bytes)
    if ($Bytes -ge 1TB) { return ('{0:N2} TB' -f ($Bytes / 1TB)) }
    if ($Bytes -ge 1GB) { return ('{0:N2} GB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N2} MB' -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ('{0:N2} KB' -f ($Bytes / 1KB)) }
    return ('{0} B' -f $Bytes)
}

function Get-SumBytes {
    # Measure-Object emits nothing for an empty set, so .Sum is a null property
    # access under StrictMode. Always hand back a long.
    param($Items, [string] $Property = 'Size')
    $m = @($Items) | Measure-Object -Property $Property -Sum
    if ($null -eq $m -or $null -eq $m.Sum) { return [long]0 }
    return [long]$m.Sum
}

function Get-WebStatusCode {
    param($ErrorRecord)
    if ($ErrorRecord.Exception -is [Net.WebException]) {
        $r = $ErrorRecord.Exception.Response
        if ($null -ne $r) { return [int]$r.StatusCode }
    }
    return 0   # transport-level failure: DNS, TLS, reset, timeout
}

function Test-Retryable {
    param([int] $StatusCode)
    # 403 is retryable here on purpose: this bucket returns AccessDenied with an empty
    # <Message> as an edge-level throttle/blip on files that are otherwise public.
    return ($StatusCode -in @(0, 403, 408, 429, 500, 502, 503, 504))
}

function Invoke-WithRetry {
    <# Runs a scriptblock, retrying transient HTTP failures with exponential backoff + jitter. #>
    param(
        [scriptblock] $Action,
        [string]      $What,
        [int]         $Retries = $MaxRetries
    )
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            return & $Action
        }
        catch {
            $code = Get-WebStatusCode $_
            if (-not (Test-Retryable $code) -or $attempt -eq $Retries) { throw }
            $delay = [math]::Min([math]::Pow(2, $attempt), 60) + (Get-Random -Minimum 0 -Maximum 3)
            Write-Log ("{0}: HTTP {1}, retry {2}/{3} in {4:N0}s" -f $What, $code, $attempt, $Retries, $delay) 'WARN'
            Start-Sleep -Seconds $delay
        }
    }
}

#endregion

#region bucket listing ------------------------------------------------------

function Get-BucketManifest {
    <# Full ListObjectsV2 enumeration, following continuation tokens. #>
    $objects = New-Object System.Collections.Generic.List[object]
    $token   = $null
    $page    = 0

    do {
        $page++
        $uri = "$BaseUrl/?list-type=2&max-keys=1000"
        if ($token) { $uri += '&continuation-token=' + [Uri]::EscapeDataString($token) }

        $xml = Invoke-WithRetry -What "list page $page" -Action {
            $req = [Net.HttpWebRequest]::Create($uri)
            $req.UserAgent = $script:UserAgent
            $req.Timeout   = 60000
            $resp = $req.GetResponse()
            try {
                $reader = New-Object IO.StreamReader($resp.GetResponseStream())
                try { [xml]$reader.ReadToEnd() } finally { $reader.Dispose() }
            }
            finally { $resp.Close() }
        }

        $result = $xml.ListBucketResult
        if ($result.PSObject.Properties.Name -contains 'Contents') {
            foreach ($c in $result.Contents) {
                $objects.Add([pscustomobject]@{
                    Key          = $c.Key
                    Size         = [long]$c.Size
                    ETag         = $c.ETag.Trim('"')
                    LastModified = $c.LastModified
                })
            }
        }

        $truncated = ($result.IsTruncated -eq 'true')
        if ($truncated) { $token = $result.NextContinuationToken } else { $token = $null }
        Write-Log ("listed page {0}: {1} objects so far" -f $page, $objects.Count)
    } while ($token)

    return $objects
}

function Split-ObjectKey {
    <# us_apt.json -> country us, type apt, format json.  Handles asp_v1.txt etc. #>
    param([string] $Key)
    $m = [regex]::Match($Key, '^(?<c>[A-Za-z0-9]{2})_(?<t>.+)\.(?<f>[^.]+)$')
    if ($m.Success) {
        return [pscustomobject]@{
            Country = $m.Groups['c'].Value.ToLower()
            Type    = $m.Groups['t'].Value.ToLower()
            Format  = $m.Groups['f'].Value.ToLower()
        }
    }
    # Anything not matching the country_type.format convention (world files, new
    # products) is kept rather than skipped, under a sentinel country.
    return [pscustomobject]@{ Country = '_other'; Type = '_other'; Format = [IO.Path]::GetExtension($Key).TrimStart('.').ToLower() }
}

#endregion

#region download ------------------------------------------------------------

function Invoke-VerifiedDownload {
    <#
        Streams $Uri to $OutFile and guarantees the whole object landed.

        The bucket will return 200 and then stop sending mid-stream. Comparing bytes
        written against Content-Length catches that; HTTP Range resumes from the cut
        instead of restarting a 40 MB transfer.

        Returns the byte count. Throws if it cannot complete within $MaxRetries.
    #>
    param(
        [string] $Uri,
        [string] $OutFile,
        [long]   $ExpectedSize = -1
    )

    if (Test-Path -LiteralPath $OutFile) { Remove-Item -LiteralPath $OutFile -Force }
    $have     = [long]0
    $expected = $ExpectedSize

    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        $resp = $null
        try {
            $req = [Net.HttpWebRequest]::Create($Uri)
            $req.UserAgent        = $script:UserAgent
            $req.Timeout          = 60000
            $req.ReadWriteTimeout = 300000
            # No AutomaticDecompression on purpose: archive the exact bytes served.
            if ($have -gt 0) { $req.AddRange([long]$have) }

            $resp = $req.GetResponse()

            # If we asked for a range and the server ignored it, start over cleanly.
            if ($have -gt 0 -and [int]$resp.StatusCode -ne 206) {
                Write-Log "server ignored Range for $Uri; restarting transfer" 'WARN'
                $have = 0
            }

            if ($expected -lt 0) {
                if ($have -gt 0) { $expected = $have + $resp.ContentLength }
                else             { $expected = $resp.ContentLength }
            }

            if ($have -gt 0) { $mode = [IO.FileMode]::Append } else { $mode = [IO.FileMode]::Create }
            $inStream  = $resp.GetResponseStream()
            $outStream = New-Object IO.FileStream($OutFile, $mode, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $buffer = New-Object byte[] (1MB)
                while ($true) {
                    $n = $inStream.Read($buffer, 0, $buffer.Length)
                    if ($n -le 0) { break }
                    $outStream.Write($buffer, 0, $n)
                    $have += $n
                }
            }
            finally {
                $outStream.Dispose()
                $inStream.Dispose()
            }

            if ($expected -lt 0 -or $have -eq $expected) { return $have }

            Write-Log ("short read: {0} of {1} bytes for {2}; resuming" -f $have, $expected, (Split-Path $Uri -Leaf)) 'WARN'
        }
        catch {
            $code = Get-WebStatusCode $_
            if (-not (Test-Retryable $code)) { throw }
            Write-Log ("HTTP {0} on {1} (attempt {2}/{3})" -f $code, (Split-Path $Uri -Leaf), $attempt, $MaxRetries) 'WARN'
        }
        finally {
            if ($null -ne $resp) { $resp.Close() }
        }

        $delay = [math]::Min([math]::Pow(2, $attempt), 60) + (Get-Random -Minimum 0 -Maximum 3)
        Start-Sleep -Seconds $delay
    }

    throw ("failed to fully download {0}: got {1} of {2} bytes after {3} attempts" -f $Uri, $have, $expected, $MaxRetries)
}

function Test-PlausibleContent {
    <# Cheap structural sanity check; Content-Length is the real guard. #>
    param([string] $Path, [string] $Format)
    if ($Format -notin @('json', 'geojson')) { return $true }
    $fs = [IO.File]::OpenRead($Path)
    try {
        if ($fs.Length -eq 0) { return $false }
        $fs.Seek(-1, [IO.SeekOrigin]::End) | Out-Null
        $last = $fs.ReadByte()
        # ndgeojson ends on '}' ; json arrays end on ']'
        return ($last -eq 0x5D -or $last -eq 0x7D)
    }
    finally { $fs.Dispose() }
}

#endregion

#region content-addressed store --------------------------------------------

function Get-BlobPath {
    param([string] $Sha256)
    return (Join-Path (Join-Path $StoreDir $Sha256.Substring(0, 2)) ($Sha256 + '.gz'))
}

function Add-Blob {
    <# Gzip $SourceFile into the store under its sha256. Returns the hash. #>
    param([string] $SourceFile)
    $sha  = (Get-FileHash -LiteralPath $SourceFile -Algorithm SHA256).Hash.ToLower()
    $dest = Get-BlobPath $sha
    if (Test-Path -LiteralPath $dest) { return $sha }   # already archived, identical bytes

    $dir = Split-Path $dest -Parent
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    $tmp = "$dest.partial"
    $in  = [IO.File]::OpenRead($SourceFile)
    try {
        $out = [IO.File]::Create($tmp)
        try {
            $gz = New-Object IO.Compression.GZipStream($out, [IO.Compression.CompressionLevel]::Optimal)
            try { $in.CopyTo($gz) } finally { $gz.Dispose() }
        }
        finally { $out.Dispose() }
    }
    finally { $in.Dispose() }

    Move-Item -LiteralPath $tmp -Destination $dest -Force
    return $sha
}

function Expand-Blob {
    param([string] $Sha256, [string] $Destination)
    $src = Get-BlobPath $Sha256
    $dir = Split-Path $Destination -Parent
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $in = [IO.File]::OpenRead($src)
    try {
        $gz = New-Object IO.Compression.GZipStream($in, [IO.Compression.CompressionMode]::Decompress)
        try {
            $out = [IO.File]::Create($Destination)
            try { $gz.CopyTo($out) } finally { $out.Dispose() }
        }
        finally { $gz.Dispose() }
    }
    finally { $in.Dispose() }
}

#endregion

#region main ----------------------------------------------------------------

$StoreDir     = Join-Path $Root 'store'
$SnapshotDir  = Join-Path $Root 'snapshots'
$LogDir       = Join-Path $Root 'logs'
$WorkDir      = Join-Path $Root '.work'

foreach ($d in @($Root, $StoreDir, $SnapshotDir, $LogDir, $WorkDir)) {
    if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
$script:LogFile = Join-Path $LogDir ((Get-Date -Format 'yyyy-MM-dd') + '.log')

Write-Log "openAIP mirror starting - root: $Root"
$runStart = Get-Date

# --- enumerate -------------------------------------------------------------
$all = Get-BucketManifest
Write-Log ("bucket holds {0} objects, {1}" -f $all.Count, (Format-Bytes (Get-SumBytes $all)))

# --- filter ----------------------------------------------------------------
$wanted = foreach ($o in $all) {
    $p = Split-ObjectKey $o.Key
    if ($Country.Count -and ($p.Country -notin $Country)) { continue }
    if ($Type.Count    -and ($p.Type    -notin $Type))    { continue }
    if ($Format.Count  -and ($p.Format  -notin $Format))  { continue }
    [pscustomobject]@{
        Key = $o.Key; Size = $o.Size; ETag = $o.ETag; LastModified = $o.LastModified
        Country = $p.Country; Type = $p.Type; Format = $p.Format
    }
}
$wanted = @($wanted)
$wantedBytes = Get-SumBytes $wanted
Write-Log ("selected {0} objects, {1}" -f $wanted.Count, (Format-Bytes $wantedBytes))

if (-not $wanted.Count) { Write-Log 'no objects matched those filters - nothing to do.' 'WARN'; return }

if ($ListOnly) {
    Write-Log 'ListOnly: nothing downloaded.'
    $wanted | Group-Object Format |
        Sort-Object @{E = { Get-SumBytes $_.Group }} -Descending |
        ForEach-Object {
            '{0,-10} {1,5} files  {2,10}' -f $_.Name, $_.Count, (Format-Bytes (Get-SumBytes $_.Group))
        }
    return
}

# --- previous snapshot (change detection) ----------------------------------
$prev = @{}
$prevFile = Get-ChildItem -LiteralPath $SnapshotDir -Filter '*.csv' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
if ($prevFile) {
    foreach ($row in (Import-Csv -LiteralPath $prevFile.FullName)) { $prev[$row.Key] = $row }
    Write-Log ("baseline: {0} ({1} entries)" -f $prevFile.Name, $prev.Count)
}
else { Write-Log 'no previous snapshot - this is a full initial sync' }

# --- sync ------------------------------------------------------------------
$snapshot   = New-Object System.Collections.Generic.List[object]
$nDown = 0; $nSkip = 0; $nFail = 0; $bytesDown = [long]0
$i = 0

foreach ($obj in $wanted) {
    $i++
    $pct = [int](100 * $i / [math]::Max($wanted.Count, 1))
    Write-Progress -Activity 'Mirroring openAIP' -Status ("{0}/{1}  {2}" -f $i, $wanted.Count, $obj.Key) -PercentComplete $pct

    # Unchanged ETag AND the blob is still in the store => reuse, no transfer.
    $reuse = $null
    if ($prev.ContainsKey($obj.Key)) {
        $p = $prev[$obj.Key]
        if ($p.ETag -eq $obj.ETag -and (Test-Path -LiteralPath (Get-BlobPath $p.Sha256))) { $reuse = $p.Sha256 }
    }

    if ($reuse) {
        $snapshot.Add([pscustomobject]@{
            Key = $obj.Key; Country = $obj.Country; Type = $obj.Type; Format = $obj.Format
            Size = $obj.Size; ETag = $obj.ETag; LastModified = $obj.LastModified; Sha256 = $reuse
        })
        $nSkip++
        continue
    }

    $tmp = Join-Path $WorkDir $obj.Key
    try {
        $got = Invoke-VerifiedDownload -Uri "$BaseUrl/$($obj.Key)" -OutFile $tmp -ExpectedSize $obj.Size

        if (-not (Test-PlausibleContent -Path $tmp -Format $obj.Format)) {
            throw "content failed structural check (truncated or malformed): $($obj.Key)"
        }

        $sha = Add-Blob -SourceFile $tmp
        $snapshot.Add([pscustomobject]@{
            Key = $obj.Key; Country = $obj.Country; Type = $obj.Type; Format = $obj.Format
            Size = $obj.Size; ETag = $obj.ETag; LastModified = $obj.LastModified; Sha256 = $sha
        })
        $nDown++
        $bytesDown += $got
        Write-Log ("fetched {0} ({1})" -f $obj.Key, (Format-Bytes $got))
    }
    catch {
        $nFail++
        Write-Log ("FAILED {0}: {1}" -f $obj.Key, $_.Exception.Message) 'ERROR'
        # Carry the previous good version forward so the snapshot stays complete.
        if ($prev.ContainsKey($obj.Key)) {
            $p = $prev[$obj.Key]
            if (Test-Path -LiteralPath (Get-BlobPath $p.Sha256)) {
                $snapshot.Add([pscustomobject]@{
                    Key = $p.Key; Country = $p.Country; Type = $p.Type; Format = $p.Format
                    Size = $p.Size; ETag = $p.ETag; LastModified = $p.LastModified; Sha256 = $p.Sha256
                })
                Write-Log ("  -> kept previous copy of {0}" -f $obj.Key) 'WARN'
            }
        }
    }
    finally {
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
    }
}
Write-Progress -Activity 'Mirroring openAIP' -Completed

# --- write snapshot --------------------------------------------------------
$snapFile = Join-Path $SnapshotDir ((Get-Date -Format 'yyyy-MM-dd') + '.csv')
$snapshot | Sort-Object Key | Export-Csv -LiteralPath $snapFile -NoTypeInformation -Encoding utf8
Write-Log "snapshot written: $snapFile"

# --- optional plain tree ---------------------------------------------------
if ($Materialize) {
    $currentDir = Join-Path $Root 'current'
    if (Test-Path -LiteralPath $currentDir) { Remove-Item -LiteralPath $currentDir -Recurse -Force }
    foreach ($row in $snapshot) { Expand-Blob -Sha256 $row.Sha256 -Destination (Join-Path (Join-Path $currentDir $row.Country) $row.Key) }
    Write-Log "materialized plain copy: $currentDir"
}

# --- retention -------------------------------------------------------------
if ($KeepSnapshots -gt 0) {
    $snaps = @(Get-ChildItem -LiteralPath $SnapshotDir -Filter '*.csv' | Sort-Object Name -Descending)
    if ($snaps.Count -gt $KeepSnapshots) {
        $snaps | Select-Object -Skip $KeepSnapshots | ForEach-Object {
            Write-Log "retiring snapshot $($_.Name)"
            Remove-Item -LiteralPath $_.FullName -Force
        }
        # GC: drop blobs no surviving snapshot references.
        $live = @{}
        foreach ($s in (Get-ChildItem -LiteralPath $SnapshotDir -Filter '*.csv')) {
            foreach ($row in (Import-Csv -LiteralPath $s.FullName)) { $live[$row.Sha256] = $true }
        }
        $freed = [long]0; $gone = 0
        foreach ($blob in (Get-ChildItem -LiteralPath $StoreDir -Recurse -Filter '*.gz' -ErrorAction SilentlyContinue)) {
            if (-not $live.ContainsKey($blob.BaseName)) {
                $freed += $blob.Length; $gone++
                Remove-Item -LiteralPath $blob.FullName -Force
            }
        }
        if ($gone) { Write-Log ("garbage-collected {0} blobs, freed {1}" -f $gone, (Format-Bytes $freed)) }
    }
}

# --- summary ---------------------------------------------------------------
$storeBytes = Get-SumBytes (Get-ChildItem -LiteralPath $StoreDir -Recurse -Filter '*.gz' -ErrorAction SilentlyContinue) 'Length'
$snapCount  = @(Get-ChildItem -LiteralPath $SnapshotDir -Filter '*.csv').Count
Write-Log ('-' * 62)
Write-Log ("downloaded {0} | reused {1} | failed {2}" -f $nDown, $nSkip, $nFail)
Write-Log ("transferred {0} in {1:N1} min" -f (Format-Bytes $bytesDown), ((Get-Date) - $runStart).TotalMinutes)
Write-Log ("archive now {0} ({1:N0}% of {2} raw) across {3} snapshot(s)" -f `
    (Format-Bytes $storeBytes), (100 * $storeBytes / [math]::Max($wantedBytes, 1)), (Format-Bytes $wantedBytes), $snapCount)

if ($nFail -gt 0) { exit 1 }

#endregion
