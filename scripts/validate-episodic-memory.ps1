#Requires -Version 7.0
param(
    [string]$Binary = "target/release/axon-uic-episodic-stream.exe",
    [UInt64]$Seed = 20261012,
    [int]$Episodes = 1536,
    [int]$Queries = 20000,
    [string]$Output = "target/episodic-memory.json",
    [string]$TrainJournal = ""
)

$ErrorActionPreference = "Stop"
$taskRoot = Split-Path -Parent $PSScriptRoot
function Resolve-TaskPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $taskRoot $Value))
}

$binaryPath = Resolve-TaskPath $Binary
$outputPath = Resolve-TaskPath $Output
$rawPath = [IO.Path]::ChangeExtension($outputPath, "raw.json")
$journalPath = if ($TrainJournal) { Resolve-TaskPath $TrainJournal } else { $null }
if (-not (Test-Path -LiteralPath $binaryPath -PathType Leaf)) { throw "Missing binary: $binaryPath" }
foreach ($path in @($outputPath, $rawPath, $journalPath)) {
    if ($path -and (Test-Path -LiteralPath $path)) { throw "Existing output is preserved: $path" }
}
[IO.Directory]::CreateDirectory((Split-Path -Parent $outputPath)) | Out-Null
$reserved = [IO.File]::Open($outputPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
$start = [Diagnostics.ProcessStartInfo]::new($binaryPath)
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.WorkingDirectory = $taskRoot
foreach ($value in @("--seed", "$Seed", "--episodes", "$Episodes", "--queries", "$Queries", "--output", $rawPath)) {
    $start.ArgumentList.Add($value)
}
if ($journalPath) {
    $start.ArgumentList.Add("--train-journal")
    $start.ArgumentList.Add($journalPath)
}
$process = [Diagnostics.Process]::new()
$process.StartInfo = $start
$timer = [Diagnostics.Stopwatch]::StartNew()
$sampledPeak = 0L
$started = $false
try {
    $process.Start() | Out-Null
    $started = $true
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    while (-not $process.WaitForExit(10)) {
        $process.Refresh()
        try { $sampledPeak = [Math]::Max($sampledPeak, $process.PeakWorkingSet64) } catch [InvalidOperationException] { }
    }
    $timer.Stop()
    $stdout.GetAwaiter().GetResult() | Out-Null
    $errors = $stderr.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) { throw "Benchmark failed ($($process.ExitCode)): $errors" }
    $metrics = Get-Content -LiteralPath $rawPath -Raw | ConvertFrom-Json
    if ($metrics.exact_matches -ne $Queries) { throw "Temporal parity failed" }
    $report = [ordered]@{
        protocol = "axon-episodic-memory-v1"
        recorded_utc = [DateTime]::UtcNow.ToString("o")
        executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $binaryPath).Hash
        executable_bytes = (Get-Item -LiteralPath $binaryPath).Length
        raw_report_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $rawPath).Hash
        elapsed_ms = [Math]::Round($timer.Elapsed.TotalMilliseconds, 3)
        sampled_peak_working_set_bytes = $sampledPeak
        journal = $journalPath
        journal_sha256 = if ($journalPath) { (Get-FileHash -Algorithm SHA256 -LiteralPath $journalPath).Hash } else { $null }
        metrics = $metrics
    }
    $json = $report | ConvertTo-Json -Depth 8
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json)
    $reserved.Write($bytes, 0, $bytes.Length)
    $reserved.Flush($true)
    $json
} finally {
    if ($started -and -not $process.HasExited) { $process.Kill(); $process.WaitForExit() }
    $process.Dispose()
    $reserved.Dispose()
}
