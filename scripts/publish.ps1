#Requires -Version 5.1
<#
.SYNOPSIS
    Publish the project repo to its private remote.

.DESCRIPTION
    Encodes the transport settings that this machine needs and that are easy to
    get wrong:

      * SSH with keepalive. HTTPS is unusable here (the configured proxy is dead
        and a direct connection is reset).
      * --no-verify. The git-lfs pre-push hook tries to authenticate and loops
        when the VPN flaps; this repo carries no LFS payload.
      * An explicit refspec. Pushing to the URL directly (not to a named remote)
        leaves no remote-tracking ref, which makes a later --force-with-lease
        fail with "stale info".

    Commits are created with -s (Signed-off-by) because the upstream PR needs it.

.EXAMPLE
    .\scripts\publish.ps1 -Message "patches: add the vllm-ascend cache dtype stack"
    .\scripts\publish.ps1 -Message "docs: ..." -Branch main -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [string]$Branch = 'main',

    [string]$Remote = 'git@github.com:Pingzii/afd-a5-env.git',

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$EnvRoot = Split-Path -Parent $PSScriptRoot

$env:GIT_SSH_COMMAND = 'ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 ' +
                       '-o ServerAliveInterval=10 -o ServerAliveCountMax=6'

function Invoke-Native {
    <# Run a native command without letting its stderr terminate the script.
       git writes progress to stderr, and with $ErrorActionPreference='Stop'
       PowerShell 5.1 raises a terminating error on that first stderr line --
       which reports a successful push as a failure. Passing '2>&1' is NOT
       enough in 5.1; the preference has to be relaxed for the call itself.
       Success is the exit code, nothing else. #>
    param([string]$Command, [string[]]$Arguments)
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = & $Command @Arguments 2>&1
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $saved
    }
    $lines | ForEach-Object { Write-Host ("    " + [string]$_) }
    return $code
}

function Resolve-RemoteName {
    <# Prefer a configured remote over a raw URL: pushing to a URL leaves the
       remote-tracking ref stale, which makes a later --force-with-lease fail
       with "stale info". #>
    param([string]$Url)
    foreach ($line in (& git remote -v 2>&1)) {
        if ([string]$line -match '^(\S+)\s+(\S+)\s+\(push\)$' -and $Matches[2] -eq $Url) {
            return $Matches[1]
        }
    }
    return $null
}

Push-Location $EnvRoot
try {
    $dirty = @(& git status --porcelain 2>&1)
    if ($dirty.Count -eq 0) {
        Write-Host "  nothing to commit"
    }
    else {
        Write-Host "  changes:"
        $dirty | ForEach-Object { Write-Host ("    " + $_) }

        if ($DryRun) { Write-Host "  -DryRun: not committing or pushing"; exit 0 }

        if ((Invoke-Native -Command 'git' -Arguments @('add', '-A')) -ne 0) {
            throw "git add failed"
        }
        if ((Invoke-Native -Command 'git' -Arguments @('commit', '-s', '-m', $Message)) -ne 0) {
            throw "git commit failed"
        }
    }

    if ($DryRun) { Write-Host "  -DryRun: not pushing"; exit 0 }

    $target = Resolve-RemoteName -Url $Remote
    if (-not $target) {
        Write-Warning ("no configured remote matches {0}; pushing the URL directly" -f $Remote)
        Write-Warning "the remote-tracking ref will stay stale -- expect --force-with-lease to complain"
        $target = $Remote
    }

    Write-Host ("  pushing HEAD -> {0} {1}" -f $target, $Branch)
    if ((Invoke-Native -Command 'git' -Arguments @('push', '--no-verify', $target, ("HEAD:refs/heads/{0}" -f $Branch))) -ne 0) {
        throw "push failed"
    }

    Write-Host ("  done: {0}" -f (& git log -1 --format='%h %s'))
}
finally {
    Pop-Location
}
