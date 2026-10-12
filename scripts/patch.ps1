#Requires -Version 5.1
<#
.SYNOPSIS
    Patch lifecycle for changes to external repos (vllm-ascend, vllm, ...).

.DESCRIPTION
    A patch set is a directory:  patches/<repo>/<name>/
      NNNN-*.patch     one file per logical change, applied in name order
      combined.patch   the same hunks concatenated, for a one-shot apply
      README.md        why / apply / revert / verification status / upstream

    Actions:
      new      Capture the source repo's dirty tree into NNNN-*.patch files,
               then round-trip verify and report. Refuses if the tree is clean.
      combine  Rebuild combined.patch from the NNNN-*.patch files.
      check    On a clean tree, confirm the stack applies (git apply --check).
      apply    Apply the stack, then print the resulting git status.
      revert   Reverse the stack, delete the files it ADDED, then assert the
               tree is clean again. Reverting is not complete until that check
               passes -- a diagnostic left behind becomes the next person's
               mystery.
      list     Show the stacks that exist, and whether each has a README.

    ASCII only: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI, so non-ASCII
    source would break parsing before a single line runs.

.EXAMPLE
    pwsh -File scripts\patch.ps1 new    -Repo vllm-ascend -Name dsv4-a5-cache-dtype
    pwsh -File scripts\patch.ps1 check  -Repo vllm-ascend -Name dsv4-a5-cache-dtype
    pwsh -File scripts\patch.ps1 apply  -Repo vllm-ascend -Name dsv4-a5-cache-dtype
    pwsh -File scripts\patch.ps1 revert -Repo vllm-ascend -Name dsv4-a5-cache-dtype
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('new', 'combine', 'check', 'apply', 'revert', 'list')]
    [string]$Action,

    [string]$Repo,

    [string]$Name,

    # 'new' only: restrict the captured diff to these paths (relative to the source repo).
    # Without it, every tracked modification in the tree is captured.
    [string[]]$Files
)

$ErrorActionPreference = 'Stop'

$EnvRoot = Split-Path -Parent $PSScriptRoot
$Workspace = Split-Path -Parent $EnvRoot

function Get-PatchDir {
    param([string]$Repository, [string]$PatchName)
    if (-not $Repository) { throw "-Repo is required" }
    if (-not $PatchName) { throw "-Name is required" }
    return (Join-Path $EnvRoot ("patches\{0}\{1}" -f $Repository, $PatchName))
}

function Get-SourceRepo {
    param([string]$Repository)
    if (-not $Repository) { throw "-Repo is required" }
    $path = Join-Path $Workspace $Repository
    if (-not (Test-Path (Join-Path $path '.git'))) {
        throw ("not a git repo: {0}" -f $path)
    }
    return $path
}

function Invoke-Git {
    param([string]$Repository, [string[]]$Arguments)
    $lines = & git -C $Repository @Arguments 2>&1
    return [pscustomobject]@{
        Code = $LASTEXITCODE
        Out  = @($lines | ForEach-Object { [string]$_ })
    }
}

function Get-StackFiles {
    param([string]$Directory)
    $files = @(Get-ChildItem -Path $Directory -Filter '*.patch' -File |
        Where-Object { $_.Name -ne 'combined.patch' } |
        Sort-Object Name)
    if ($files.Count -eq 0) {
        throw ("no NNNN-*.patch files in {0}" -f $Directory)
    }
    return $files
}

function Get-AddedPaths {
    <# Paths the patch CREATES. 'git apply -R' restores modified files but leaves
       these behind, so revert has to delete them explicitly. #>
    param([string]$PatchPath)
    $added = New-Object System.Collections.Generic.List[string]
    $current = $null
    foreach ($line in (Get-Content -LiteralPath $PatchPath)) {
        if ($line -like 'diff --git *') {
            $current = $null
        }
        elseif ($line -like '+++ b/*') {
            $current = $line.Substring(6)
        }
        elseif ($line -like 'new file mode*' -and $current) {
            [void]$added.Add($current)
        }
    }
    return $added
}

function Write-Status {
    param([string]$Repository)
    $status = Invoke-Git -Repository $Repository -Arguments @('status', '--porcelain')
    if ($status.Out.Count -eq 0) {
        Write-Host "  worktree clean"
    }
    else {
        $status.Out | ForEach-Object { Write-Host ("  " + $_) }
    }
    return $status.Out.Count
}

# ---------------------------------------------------------------- list

if ($Action -eq 'list') {
    $patchesRoot = Join-Path $EnvRoot 'patches'
    if (-not (Test-Path $patchesRoot)) { Write-Host "no patches/ directory"; exit 0 }
    Get-ChildItem -Path $patchesRoot -Directory | ForEach-Object {
        $repository = $_.Name
        Get-ChildItem -Path $_.FullName -Directory | ForEach-Object {
            $stack = $_.FullName
            $count = @(Get-ChildItem -Path $stack -Filter '*.patch' -File |
                Where-Object { $_.Name -ne 'combined.patch' }).Count
            $hasReadme = Test-Path (Join-Path $stack 'README.md')
            $mark = 'OK '
            if (-not $hasReadme) { $mark = 'MISSING-README' }
            Write-Host ("  {0}  {1}/{2}  ({3} patch(es))" -f $mark, $repository, $_.Name, $count)
        }
    }
    exit 0
}

# ---------------------------------------------------------------- new

if ($Action -eq 'new') {
    $source = Get-SourceRepo -Repository $Repo
    $target = Get-PatchDir -Repository $Repo -PatchName $Name
    if (Test-Path $target) { throw ("already exists: {0}" -f $target) }

    $diffArgs = @('diff', '--no-color')
    if ($Files) { $diffArgs += '--'; $diffArgs += $Files }
    $diff = Invoke-Git -Repository $source -Arguments $diffArgs
    if ($diff.Out.Count -eq 0) {
        throw "the source repo has no tracked modifications -- nothing to capture"
    }

    # Untracked files are NOT in 'git diff'. Refuse rather than capture a partial change.
    $untracked = Invoke-Git -Repository $source -Arguments @('ls-files', '--others', '--exclude-standard')
    if ($untracked.Out.Count -gt 0) {
        Write-Warning ("untracked files are not captured by 'git diff':")
        $untracked.Out | ForEach-Object { Write-Warning ("  " + $_) }
        Write-Warning "stage them with 'git add -N' first, or they will be missing from the patch."
    }

    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $patchFile = Join-Path $target '0001-change.patch'
    # LF endings: 'git apply' matches context lines byte for byte.
    $text = ($diff.Out -join "`n") + "`n"
    [System.IO.File]::WriteAllText($patchFile, $text, (New-Object System.Text.UTF8Encoding $false))
    Write-Host ("  wrote {0}" -f $patchFile)

    $readme = @"
# $Name

**Target repo**: `$Repo`
**Base**: ``$(($(Invoke-Git -Repository $source -Arguments @('rev-parse', '--short', 'HEAD')).Out))``

## Why

<!-- what breaks without this, with the error text or the measured evidence -->

## Files

$(($diff.Out | Where-Object { $_ -like 'diff --git *' } | ForEach-Object { "- ``" + ($_ -replace '^diff --git a/', '' -replace ' b/.*$', '') + "``" }) -join "`n")

## Apply / revert

``````bash
P=patches/$Repo/$Name
git -C <$Repo> apply --check `$P/0001-change.patch
git -C <$Repo> apply `$P/0001-change.patch
git -C <$Repo> apply -R `$P/0001-change.patch
``````

## Verification status

- [ ] applies cleanly on the pinned base (`patch.ps1 check`)
- [ ] service starts
- [ ] output matches the baseline byte for byte <- the one that gets skipped
- [ ] upstream status recorded below

## Upstream

<!-- has someone already fixed this? link the issue/PR and say what it covers -->
"@
    [System.IO.File]::WriteAllText((Join-Path $target 'README.md'), $readme, (New-Object System.Text.UTF8Encoding $false))
    Write-Host ("  wrote {0}" -f (Join-Path $target 'README.md'))

    Write-Host "  now fill in README.md, then:  patch.ps1 combine -Repo $Repo -Name $Name"
    exit 0
}

# ---------------------------------------------------------------- combine / check / apply / revert

$patchDir = Get-PatchDir -Repository $Repo -PatchName $Name
if (-not (Test-Path $patchDir)) { throw ("no such patch set: {0}" -f $patchDir) }
$sourceRepo = Get-SourceRepo -Repository $Repo
$stack = Get-StackFiles -Directory $patchDir
$combined = Join-Path $patchDir 'combined.patch'

if ($Action -eq 'combine') {
    $parts = @()
    foreach ($f in $stack) {
        $parts += ([System.IO.File]::ReadAllText($f.FullName)).TrimEnd("`r", "`n")
    }
    [System.IO.File]::WriteAllText($combined, (($parts -join "`n") + "`n"), (New-Object System.Text.UTF8Encoding $false))
    Write-Host ("  wrote {0} ({1} patch(es))" -f $combined, $stack.Count)
    exit 0
}

if ($Action -eq 'check' -or $Action -eq 'apply') {
    $dirty = Write-Status -Repository $sourceRepo
    if ($dirty -gt 0) {
        throw "the source repo is not clean -- revert before checking or applying"
    }

    $failed = $false
    foreach ($f in $stack) {
        $r = Invoke-Git -Repository $sourceRepo -Arguments @('apply', '--check', $f.FullName)
        $verdict = 'ok'
        if ($r.Code -ne 0) { $verdict = 'FAILED'; $failed = $true }
        Write-Host ("  [{0}] {1}" -f $verdict, $f.Name)
        if ($r.Code -ne 0) { $r.Out | ForEach-Object { Write-Host ("        " + $_) } }
    }
    if (Test-Path $combined) {
        $r = Invoke-Git -Repository $sourceRepo -Arguments @('apply', '--check', $combined)
        $verdict = 'ok'
        if ($r.Code -ne 0) { $verdict = 'FAILED'; $failed = $true }
        Write-Host ("  [{0}] combined.patch" -f $verdict)
    }
    if ($failed) { throw "the stack does not apply to the current base" }
    if ($Action -eq 'check') { Write-Host "  check passed"; exit 0 }

    foreach ($f in $stack) {
        $r = Invoke-Git -Repository $sourceRepo -Arguments @('apply', $f.FullName)
        if ($r.Code -ne 0) { throw ("apply failed: {0}" -f $f.Name) }
        Write-Host ("  applied {0}" -f $f.Name)
    }
    Write-Host "  resulting worktree:"
    [void](Write-Status -Repository $sourceRepo)
    Write-Host ""
    Write-Host "  next: run the experiment, then 'patch.ps1 revert' and confirm the tree is clean."
    exit 0
}

if ($Action -eq 'revert') {
    # Reverse in the opposite order, then delete what the stack created.
    for ($i = $stack.Count - 1; $i -ge 0; $i--) {
        $f = $stack[$i]
        $r = Invoke-Git -Repository $sourceRepo -Arguments @('apply', '-R', $f.FullName)
        if ($r.Code -ne 0) {
            Write-Warning ("reverse apply failed (already reverted?): {0}" -f $f.Name)
            $r.Out | ForEach-Object { Write-Warning ("  " + $_) }
        }
        else {
            Write-Host ("  reverted {0}" -f $f.Name)
        }
        foreach ($rel in (Get-AddedPaths -PatchPath $f.FullName)) {
            $abs = Join-Path $sourceRepo $rel
            if (Test-Path $abs) {
                Remove-Item -LiteralPath $abs -Force
                Write-Host ("  removed added file {0}" -f $rel)
            }
        }
    }

    $left = Write-Status -Repository $sourceRepo
    if ($left -gt 0) {
        Write-Host ""
        Write-Warning "the worktree is NOT clean -- something the stack did is still there."
        Write-Warning "resolve it by hand; do not start the next experiment on top of it."
        exit 1
    }
    Write-Host "  revert complete, worktree clean"
    exit 0
}
