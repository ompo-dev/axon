param(
    [ValidateRange(1024,65535)][int]$Port=4319,
    [ValidateRange(100,10000)][int]$Count=10000,
    [ValidateRange(1,100)][int]$MaxBatches=4,
    [ValidateRange(1,2147483647)][int]$Seed=61423917
)
$ErrorActionPreference='Stop'
$runs=@()
for ($batch=0; $batch -lt $MaxBatches; $batch++) {
    $raw = & (Join-Path $PSScriptRoot 'run-observatory-workflow.ps1') -Port $Port -Mode nursery -Count $Count -Seed $Seed | Out-String
    $run = $raw | ConvertFrom-Json
    $runs += [ordered]@{id=$run.id;outcome=$run.result.outcome;new=$run.result.new_candidates;unique=$run.result.unique;correct=$run.result.correct;checked=$run.result.checked;digits=$run.result.genes;admitted=$run.result.admitted}
    Write-Host ("Batch {0}: {1}; {2} unique; best {3}/{4}; {5} digits" -f ($batch+1),$run.result.outcome,$run.result.unique,$run.result.correct,$run.result.checked,$run.result.genes)
    if ($run.result.outcome -ne 'batch-limit') { break }
}
[ordered]@{runs=$runs;budget_exhausted=($runs[-1].outcome -eq 'batch-limit');all_tests_passed=($runs[-1].admitted -eq $true)} | ConvertTo-Json -Depth 6
