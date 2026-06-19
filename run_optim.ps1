#!/usr/bin/env pwsh
# Run CMA-ES optimisation for all UQ methods (5 generations, population of 4).
# Run with:
#   ./run_optim.ps1
#
# Set $Shutdown = 0 below to keep the machine on after the run.
# On Windows, shutdown is attempted with Stop-Computer first, then shutdown.exe.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PopSize = 4
$NGen = 5
$DropoutIters = 10
$BaseOut = 'results/optim'
$Shutdown = 1

$script:ExitCode = 0

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Host "===== $Title ====="
    & $Command

    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Title' failed with exit code $LASTEXITCODE"
    }
}

try {
    Invoke-Step -Title 'Temperature Scaling' -Command {
     }

    Invoke-Step -Title 'Monte Carlo Dropout' -Command {
        python optimize.py -e mcd -p $PopSize -n $NGen -i $DropoutIters -o "$BaseOut/mcd"
    }

    Invoke-Step -Title 'Levenshtein Monte Carlo Dropout' -Command {
        python optimize.py -e lmcd -p $PopSize -n $NGen -i $DropoutIters -o "$BaseOut/lmcd"
    }

    Write-Host '===== All optimisations complete ====='
}
catch {
    $script:ExitCode = 1
    Write-Host "===== Script exited with error: $($_.Exception.Message) ====="
}
finally {
    if ($Shutdown -eq 1) {
        Write-Host '===== Powering off in 60s (Ctrl-C to cancel) ====='
        Start-Sleep -Seconds 60

        try {
            Stop-Computer -Force -ErrorAction Stop
        }
        catch {
            shutdown.exe /s /t 0 | Out-Null
        }
    }

    exit $script:ExitCode
}
