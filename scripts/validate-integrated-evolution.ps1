param([ValidateRange(1024,65535)][int]$Port = 4317)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$file = Join-Path $root '.axon-memory/evolution/integrated-reserve-20261014-v1.json'
if (Test-Path -LiteralPath $file) { throw 'Frozen protocol already recorded; do not overwrite or reuse this reserve for tuning.' }
$runs = @()
foreach ($seed in @(39158217,76019431,24863959)) {
    foreach ($strategy in @('hybrid','mutation-only','balanced')) {
        $result = (& (Join-Path $PSScriptRoot 'run-observatory-workflow.ps1') -Port $Port -Mode evolution -Seed $seed -Count 10000 -Strategy $strategy -Curriculum transfer -ResumeCompleted | ConvertFrom-Json)
        $runs += $result
        [ordered]@{seed=$seed;strategy=$strategy;id=$result.id;passed=$result.result.passed;correct=$result.result.correct;heldout=$result.result.heldout;gaps=$result.trace.gaps} | ConvertTo-Json -Compress
    }
}
$report = [ordered]@{schema='axon-integrated-reserve-v1';heldout_per_task=10000;tasks=8;max_work_per_task=100000000;telemetry='search-checkpoints';rss=$null;watts=$null;runs=$runs}
$stream = [System.IO.File]::Open($file,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
try { $bytes=[System.Text.Encoding]::UTF8.GetBytes(($report | ConvertTo-Json -Depth 15)); $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
"Recorded $file"
