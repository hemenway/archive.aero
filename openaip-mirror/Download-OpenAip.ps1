<#
.SYNOPSIS
    Simple openAIP downloader - copies the daily export files onto this card as plain files.

.DESCRIPTION
    Lists the public openAIP export bucket (no API key, no account) and downloads every
    matching file to <Out>\<country>\<filename>, e.g. data\us\us_apt.geojson.

    GeoJSON only by default - it carries every field the API has and opens directly in
    QGIS, ogr2ogr, Leaflet and most EFB importers. Other formats via -Format.

    Re-runs are cheap: a file is skipped when the copy on the card already has the
    server's size and Last-Modified. Every download is checked against Content-Length
    and resumed with HTTP Range until complete, because this bucket sometimes answers
    200 and then stops sending bytes mid-file.

.EXAMPLE
    .\Download-OpenAip.ps1 -ListOnly
    Show what would be downloaded and how big it is.

.EXAMPLE
    .\Download-OpenAip.ps1
    All countries, GeoJSON, into F:\openaip-mirror\data.

.EXAMPLE
    .\Download-OpenAip.ps1 -Country us,ca -Format geojson,txt
    North America; GeoJSON plus OpenAIR airspace files.
#>
[CmdletBinding()]
param(
    [string]   $Out        = (Join-Path $PSScriptRoot 'data'),
    [string[]] $Country    = @(),            # e.g. us,ca,mx      (default: all)
    [string[]] $Type       = @(),            # apt,asp,nav,obs,rpp,hgl,hot,rca,raa (default: all)
    [string[]] $Format     = @('geojson'),   # json,geojson,ndgeojson,cup,cupx,txt,xml, or 'all'
    [switch]   $ListOnly,
    [switch]   $Force,
    [int]      $MaxRetries = 6,
    [string]   $BaseUrl    = 'https://storage.openaip.net/openaip-system-exports'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$UserAgent = 'openaip-simple-downloader/1.0'

function Format-Bytes([long] $Bytes) {
    if ($Bytes -ge 1GB) { return ('{0:N2} GB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N1} MB' -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ('{0:N0} KB' -f ($Bytes / 1KB)) }
    return "$Bytes B"
}

# ---- 1. list the bucket (S3 ListObjectsV2, paged with continuation tokens) ----------

function Get-BucketObjects {
    $objects = @()
    $token = $null
    do {
        $url = "$BaseUrl/?list-type=2&max-keys=1000"
        if ($token) { $url += '&continuation-token=' + [Uri]::EscapeDataString($token) }
        $resp = Invoke-WebRequest -Uri $url -UserAgent $UserAgent -UseBasicParsing -TimeoutSec 60
        [xml] $xml = $resp.Content
        $ns = New-Object Xml.XmlNamespaceManager $xml.NameTable
        $ns.AddNamespace('s3', 'http://s3.amazonaws.com/doc/2006-03-01/')
        foreach ($c in $xml.SelectNodes('//s3:Contents', $ns)) {
            $objects += [pscustomobject]@{
                Key          = $c.Key
                Size         = [long] $c.Size
                LastModified = ([DateTimeOffset]::Parse($c.LastModified)).UtcDateTime
            }
        }
        $trunc = $xml.SelectSingleNode('//s3:IsTruncated', $ns)
        $next  = $xml.SelectSingleNode('//s3:NextContinuationToken', $ns)
        $token = if ($trunc -and $trunc.InnerText -eq 'true' -and $next) { $next.InnerText } else { $null }
    } while ($token)
    return $objects
}

# ---- 2. download one file, verifying length and resuming with Range ----------------

function Get-BucketFile([string] $Key, [string] $Dest, [long] $Expected) {
    $tmp = "$Dest.part"
    $url = "$BaseUrl/$Key"
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        $have = if (Test-Path -LiteralPath $tmp) { (Get-Item -LiteralPath $tmp).Length } else { 0L }
        if ($have -eq $Expected) { break }
        if ($have -gt $Expected) { Remove-Item -LiteralPath $tmp; $have = 0L }   # can't be right; start over
        if ($attempt -gt 1) { Start-Sleep -Seconds ([Math]::Min(60, [Math]::Pow(2, $attempt))) }
        try {
            $req = [Net.HttpWebRequest]::Create($url)
            $req.UserAgent = $UserAgent
            $req.Timeout = 30000
            $req.ReadWriteTimeout = 60000
            if ($have -gt 0) { $req.AddRange([long] $have) }
            $resp = $req.GetResponse()
            try {
                $mode = 'Create'
                if ($have -gt 0) {
                    if ([int] $resp.StatusCode -eq 206) { $mode = 'Append' }
                    else { $have = 0L }                                        # server ignored Range
                }
                $in = $resp.GetResponseStream()
                $fs = [IO.File]::Open($tmp, $mode, 'Write', 'None')
                try { $in.CopyTo($fs, 1MB) } finally { $fs.Dispose(); $in.Dispose() }
            } finally { $resp.Close() }
        } catch [Net.WebException] {
            $r = $_.Exception.Response
            if ($r -and [int] $r.StatusCode -eq 404) { throw "404 not found" }  # listed but gone; no point retrying
            Write-Host ("      retry {0}/{1}: {2}" -f $attempt, $MaxRetries, $_.Exception.Message) -ForegroundColor Yellow
        } catch [IO.IOException] {
            Write-Host ("      retry {0}/{1}: {2}" -f $attempt, $MaxRetries, $_.Exception.Message) -ForegroundColor Yellow
        }
    }
    $have = if (Test-Path -LiteralPath $tmp) { (Get-Item -LiteralPath $tmp).Length } else { 0L }
    if ($have -ne $Expected) { throw "incomplete after $MaxRetries attempts ($have of $Expected bytes)" }
    Move-Item -LiteralPath $tmp -Destination $Dest -Force
}

# ---- 3. main -----------------------------------------------------------------------

Write-Host "Listing $BaseUrl ..."
$all = Get-BucketObjects

# Keys look like  us_apt.geojson  or  de_asp_v2.txt
$rx = '^(?<country>[a-z]{2})_(?<type>[a-z]+)(?:_v(?<ver>\d))?\.(?<ext>[a-z]+)$'
$wanted = @()
foreach ($o in $all) {
    if ($o.Key -notmatch $rx) { continue }
    if ($Country.Count -and $Matches.country -notin $Country)         { continue }
    if ($Type.Count    -and $Matches.type    -notin $Type)            { continue }
    if ($Format -notcontains 'all' -and $Matches.ext -notin $Format)  { continue }
    $o | Add-Member -NotePropertyName Country -NotePropertyValue $Matches.country
    $wanted += $o
}
$totalBytes = [long] 0; foreach ($o in $wanted) { $totalBytes += $o.Size }
Write-Host ("{0} objects in bucket; {1} match ({2})" -f $all.Count, $wanted.Count, (Format-Bytes $totalBytes))
if (-not $wanted.Count) { Write-Host 'Nothing matches the filters.'; exit 1 }

if ($ListOnly) {
    $wanted | Sort-Object Key | ForEach-Object { '{0,12}  {1}' -f (Format-Bytes $_.Size), $_.Key }
    exit 0
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$downloaded = 0; $skipped = 0; $failed = @(); $bytes = [long] 0
$i = 0
foreach ($o in $wanted) {
    $i++
    $dir  = Join-Path $Out $o.Country
    $dest = Join-Path $dir $o.Key
    $tag  = '[{0}/{1}] {2,-24} {3,10}' -f $i, $wanted.Count, $o.Key, (Format-Bytes $o.Size)

    if (-not $Force -and (Test-Path -LiteralPath $dest)) {
        $f = Get-Item -LiteralPath $dest
        # exFAT/FAT keep coarse timestamps, so allow a little slack on the comparison.
        if ($f.Length -eq $o.Size -and $f.LastWriteTimeUtc -ge $o.LastModified.AddSeconds(-2)) {
            Write-Host "$tag  up to date" -ForegroundColor DarkGray
            $skipped++; continue
        }
    }

    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "$tag  downloading"
    try {
        Get-BucketFile -Key $o.Key -Dest $dest -Expected $o.Size
        (Get-Item -LiteralPath $dest).LastWriteTimeUtc = $o.LastModified
        $downloaded++; $bytes += $o.Size
    } catch {
        Write-Host "$tag  FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $failed += $o.Key
    }
}

Write-Host ''
Write-Host ("Done. {0} downloaded ({1}), {2} up to date, {3} failed." -f $downloaded, (Format-Bytes $bytes), $skipped, $failed.Count)
if ($failed.Count) { $failed | ForEach-Object { Write-Host "  failed: $_" -ForegroundColor Red }; exit 1 }
exit 0
