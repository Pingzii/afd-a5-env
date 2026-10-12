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

Push-Location $EnvRoot
try {
    $dirty = @(git status --porcelain)
    if ($dirty.Count -eq 0) {
        Write-Host "  nothing to commit"
    }
    else {
        Write-Host "  staged changes:"
        $dirty | ForEach-Object { Write-Host ("    " + $_) }

        if ($DryRun) {
            Write-Host "  -DryRun: not committing or pushing"
            exit 0
        }

        git add -A
        git commit -s -m $Message | ForEach-Object { Write-Host ("  " + $_) }
        if ($LASTEXITCODE -ne 0) { throw "commit failed" }
    }

    if ($DryRun) {
        Write-Host "  -DryRun: not pushing"
        exit 0
    }

    Write-Host ("  pushing HEAD -> {0} {1}" -f $Remote, $Branch)
    git push --no-verify $Remote ("HEAD:refs/heads/{0}" -f $Branch)
    if ($LASTEXITCODE -ne 0) { throw "push failed" }

    Write-Host ("  done: {0}" -f (git log -1 --format='%h %s'))
}
finally {
    Pop-Location
}
