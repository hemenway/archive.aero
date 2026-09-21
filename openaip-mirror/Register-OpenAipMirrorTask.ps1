<#
.SYNOPSIS
    Registers a Windows Scheduled Task that runs the openAIP mirror daily.

.DESCRIPTION
    openAIP rebuilds its export bucket once a day; the objects observed on
    2026-09-17 carried a Last-Modified of about 03:14 UTC. The default run time
    here is 06:00 local, which clears that rebuild for any US time zone.

    A run where nothing changed costs one bucket listing and a few seconds, so
    running more often than strictly necessary is cheap. Add a second trigger if
    you would rather catch the rebuild sooner.

    This script only REGISTERS the task - it does not change the mirror itself.
    Re-running it updates the existing task in place.

.PARAMETER Root
    Archive root passed through to Sync-OpenAip.ps1.

.PARAMETER At
    Daily start time, local. Default 06:00.

.PARAMETER TaskName
    Scheduled task name. Default 'openAIP Mirror'.

.PARAMETER ExtraArgs
    Additional arguments for Sync-OpenAip.ps1, e.g. -ExtraArgs '-Country us,ca -KeepSnapshots 90'.

.PARAMETER RunWhenLoggedOff
    Register with S4U so the task runs even when you are signed out. Requires an
    elevated shell. Without this the task runs only while you are logged on.

.PARAMETER Unregister
    Remove the task instead of creating it.

.EXAMPLE
    .\Register-OpenAipMirrorTask.ps1 -Root D:\Archives\openaip

.EXAMPLE
    .\Register-OpenAipMirrorTask.ps1 -Root D:\Archives\openaip -At 04:30 -ExtraArgs '-KeepSnapshots 90'
#>
[CmdletBinding()]
param(
    [string] $Root     = (Join-Path $PSScriptRoot 'archive'),
    [string] $At       = '06:00',
    [string] $TaskName = 'openAIP Mirror',
    [string] $ExtraArgs = '',
    [switch] $RunWhenLoggedOff,
    [switch] $Unregister
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "removed scheduled task '$TaskName'" -ForegroundColor Green
    return
}

$syncScript = Join-Path $PSScriptRoot 'Sync-OpenAip.ps1'
if (-not (Test-Path -LiteralPath $syncScript)) { throw "Sync-OpenAip.ps1 not found next to this script" }

$argLine = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Root "{1}" {2}' -f $syncScript, $Root, $ExtraArgs

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argLine.Trim()
$trigger = New-ScheduledTaskTrigger -Daily -At $At

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 15)

# Don't fight the laptop: skip the run on battery rather than half-syncing.
$settings.DisallowStartIfOnBatteries = $true
$settings.StopIfGoingOnBatteries     = $true

if ($RunWhenLoggedOff) {
    # S4U runs without storing a password, but registering it needs elevation.
    $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
                                            -LogonType S4U -RunLevel Limited
}
else {
    $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
                                            -LogonType Interactive -RunLevel Limited
}

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Set-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null
    Write-Host "updated scheduled task '$TaskName'" -ForegroundColor Green
}
else {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
        -Description 'Mirrors the openAIP export bucket into a local de-duplicated archive.' | Out-Null
    Write-Host "created scheduled task '$TaskName'" -ForegroundColor Green
}

Write-Host "  runs daily at $At"
Write-Host "  archive root: $Root"
if (-not $RunWhenLoggedOff) { Write-Host "  note: runs only while you are logged on (use -RunWhenLoggedOff from an elevated shell to change)" -ForegroundColor Yellow }
Write-Host "`nrun it once now with:  Start-ScheduledTask -TaskName '$TaskName'"
