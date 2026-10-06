# Run a resumable vesselnet job until it succeeds, on a machine shared with other memory-hungry jobs.
#
#   powershell -File vesselnet/resume.ps1 -Log ..\logs\job.log -FreeGB 12 -Cmd "vesselnet/gen.py --split test --out data"
#
# Before each attempt it waits until at least -FreeGB GB of commit charge is free (Windows refuses allocations
# at the commit limit however much RAM looks free), then runs the venv's python with the given arguments
# from the LIMBUS checkout, appending stdout and stderr to -Log. It stops when the job exits 0 or after
# -MaxTries attempts. Every attempt's start, free commit and exit code go to the log.
param(
    [Parameter(Mandatory = $true)][string]$Log,
    [double]$FreeGB = 12,
    [int]$MaxTries = 30,
    [Parameter(Mandatory = $true)][string]$Cmd       # the python arguments, separated by spaces
)
$PyArgs = $Cmd -split ' +' 
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path (Split-Path -Parent $repo) ".venv\Scripts\python.exe"
$env:PYTHONPATH = $repo
foreach ($v in "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS") {
    if (-not (Test-Path "env:$v")) { Set-Item "env:$v" "4" }
}
function Free-Commit {
    $c = Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit'
    ($c.CounterSamples[1].CookedValue - $c.CounterSamples[0].CookedValue) / 1GB
}
for ($try = 1; $try -le $MaxTries; $try++) {
    while ((Free-Commit) -lt $FreeGB) { Start-Sleep -Seconds 60 }
    "[{0}] attempt {1}: {2:N1} GB commit free; python {3}" -f (Get-Date -Format s), $try, (Free-Commit), ($PyArgs -join ' ') |
        Out-File -Append -Encoding utf8 $Log
    $p = Start-Process -FilePath $py -ArgumentList $PyArgs -WorkingDirectory $repo -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$Log.out" -RedirectStandardError "$Log.err"
    Get-Content "$Log.out", "$Log.err" | Out-File -Append -Encoding utf8 $Log
    "[{0}] attempt {1} exited {2}" -f (Get-Date -Format s), $try, $p.ExitCode | Out-File -Append -Encoding utf8 $Log
    if ($p.ExitCode -eq 0) { break }
    Start-Sleep -Seconds 120
}
