param(
    [string]$OutputBase = '.ecc/benchmarks/world-resource-dev-20261012-v1',
    [uint64]$Seed = 317031,
    [int]$Worlds = 144
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'target/release/axon-uic-world-lab.exe'
$base = [System.IO.Path]::GetFullPath($OutputBase, $root)
$csv = "$base.csv"
$stdout = "$base.stdout.txt"
$stderr = "$base.stderr.txt"
$report = "$base.json"
foreach ($path in @($csv, $stdout, $stderr, $report)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing existing output: $path" }
}
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the release world lab first.' }
if ($Worlds -lt 1 -or $Worlds -gt 256) { throw 'Worlds must be 1..256.' }
$parent = Split-Path -Parent $base
[System.IO.Directory]::CreateDirectory($parent) | Out-Null
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $exe -ArgumentList @("$Seed", "$Worlds", "`"$csv`"") -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$sampledWorkingSet = 0L
$sampledPrivate = 0L
$cpuMs = 0.0
$samples = 0
while (-not $process.HasExited) {
    $process.Refresh()
    if (-not $process.HasExited) {
        $sampledWorkingSet = [Math]::Max($sampledWorkingSet, $process.WorkingSet64)
        $sampledPrivate = [Math]::Max($sampledPrivate, $process.PrivateMemorySize64)
        $cpuMs = [Math]::Max($cpuMs, $process.TotalProcessorTime.TotalMilliseconds)
        $samples++
    }
    Start-Sleep -Milliseconds 2
}
$process.WaitForExit()
$watch.Stop()
if ($process.ExitCode -ne 0) { throw (Get-Content -LiteralPath $stderr -Raw) }
$rows = @(Import-Csv -LiteralPath $csv)
if ($rows.Count -ne $Worlds * 3) { throw 'Incomplete resource lab report.' }
[ordered]@{
    seed = $Seed
    worlds = $Worlds
    executable_bytes = (Get-Item -LiteralPath $exe).Length
    sampled_working_set_bytes = $sampledWorkingSet
    sampled_private_bytes = $sampledPrivate
    sampled_cpu_ms = $cpuMs
    wall_ms = $watch.Elapsed.TotalMilliseconds
    samples = $samples
    requested_interval_ms = 2
    measurement_scope = 'One release lab process, three policies plus audits; no journal. Sampled maxima are lower bounds, not guaranteed peaks. CPU excludes final unsampled interval and the PowerShell host.'
    raw_csv = $csv
    raw_sha256 = (Get-FileHash -LiteralPath $csv -Algorithm SHA256).Hash
} | ConvertTo-Json | Out-File -LiteralPath $report -Encoding utf8NoBOM -NoClobber
Get-Content -LiteralPath $report
