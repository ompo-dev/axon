param(
    [ValidateSet('demo', 'lab')][string]$Mode = 'demo',
    [string]$OutputBase = '',
    [int]$Systems = 32
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutputBase) { $OutputBase = ".ecc/benchmarks/behavior-resource-$Mode-20261013-v1" }
$base = [System.IO.Path]::GetFullPath($OutputBase, $root)
$report = "$base.json"
$stdout = "$base.stdout.txt"
$stderr = "$base.stderr.txt"
$data = if ($Mode -eq 'demo') { "$base.outputs" } else { "$base.csv" }
foreach ($path in @($report, $stdout, $stderr, $data)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing existing output: $path" }
}
if ($Systems -lt 1 -or $Systems -gt 128) { throw 'Systems must be 1..128.' }
$exe = Join-Path $root $(if ($Mode -eq 'demo') { 'target/release/axon.exe' } else { 'target/release/axon-uic-behavior-lab.exe' })
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the release binaries first.' }
[System.IO.Directory]::CreateDirectory((Split-Path -Parent $base)) | Out-Null
$arguments = if ($Mode -eq 'demo') { @('behavior-demo', "`"$data`"") } else { @('513013', "$Systems", "`"$data`"") }
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$workingSet = 0L
$privateBytes = 0L
$cpuMs = 0.0
$samples = 0
while (-not $process.HasExited) {
    try {
        $process.Refresh()
        if (-not $process.HasExited) {
            $workingSet = [Math]::Max($workingSet, $process.WorkingSet64)
            $privateBytes = [Math]::Max($privateBytes, $process.PrivateMemorySize64)
            $cpuMs = [Math]::Max($cpuMs, $process.TotalProcessorTime.TotalMilliseconds)
            $samples++
        }
    } catch {
        if (-not $process.HasExited) { throw }
    }
    Start-Sleep -Milliseconds 2
}
$process.WaitForExit()
$timer.Stop()
if ($process.ExitCode -ne 0) { throw (Get-Content -LiteralPath $stderr -Raw) }
$files = if ($Mode -eq 'demo') { Get-ChildItem -LiteralPath $data -File } else { Get-Item -LiteralPath $data }
$artifacts = @($files | ForEach-Object { [ordered]@{name=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} })
[ordered]@{
    mode = $Mode
    executable_bytes = (Get-Item -LiteralPath $exe).Length
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    sampled_working_set_bytes = $workingSet
    sampled_private_bytes = $privateBytes
    sampled_cpu_ms = $cpuMs
    wall_ms = $timer.Elapsed.TotalMilliseconds
    samples = $samples
    requested_interval_ms = 2
    scope = 'One Rust process; sampled maxima are lower bounds. Demo includes durable query learning, replay, exact audits and publication. Lab includes generators and auditors, not an episode journal. CPU excludes final unsampled interval; wrapper time includes process launch/wait.'
    data = $data
    artifacts = $artifacts
} | ConvertTo-Json -Depth 6 | Out-File -LiteralPath $report -NoClobber -Encoding utf8NoBOM
Get-Content -LiteralPath $report
