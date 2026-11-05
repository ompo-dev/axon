param(
    [ValidateSet('demo','replay')][string]$Mode = 'demo',
    [string]$Bundle = '.axon-memory/transfer-demo-20261013-v1',
    [string]$OutputBase = '.ecc/benchmarks/transfer-resource-20261013-v1'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$base = [System.IO.Path]::GetFullPath($OutputBase,$root)
$report = "$base.json"
$stdout = "$base.stdout.txt"
$stderr = "$base.stderr.txt"
$data = if ($Mode -eq 'demo') { "$base.outputs" } else { [System.IO.Path]::GetFullPath($Bundle,$root) }
foreach ($path in @($report,$stdout,$stderr)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing existing output: $path" }
}
if ($Mode -eq 'demo' -and (Test-Path -LiteralPath $data)) { throw "Refusing existing output: $data" }
if ($Mode -eq 'replay' -and -not (Test-Path -LiteralPath $data -PathType Container)) { throw 'Replay requires an existing demo.' }
$exe = Join-Path $root 'target/release/axon.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the release CLI first.' }
$exeHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
$payload = Join-Path $root 'examples/behavior_document.md'
$payloadHash = (Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash
$arguments = @("transfer-$Mode","`"$data`"")
if ($Mode -eq 'demo') { $arguments += @('--payload',"`"$payload`"") }
[System.IO.Directory]::CreateDirectory((Split-Path -Parent $base)) | Out-Null
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$ws = 0L
$privateBytes = 0L
$cpuMs = 0.0
$samples = 0
while (-not $process.HasExited) {
    try {
        $process.Refresh()
        if (-not $process.HasExited) {
            $ws = [Math]::Max($ws,$process.WorkingSet64)
            $privateBytes = [Math]::Max($privateBytes,$process.PrivateMemorySize64)
            $cpuMs = [Math]::Max($cpuMs,$process.TotalProcessorTime.TotalMilliseconds)
            $samples++
        }
    } catch { if (-not $process.HasExited) { throw } }
    Start-Sleep -Milliseconds 2
}
$process.WaitForExit()
$timer.Stop()
$files = if (Test-Path -LiteralPath $data -PathType Container) { @(Get-ChildItem -LiteralPath $data -File) } else { @() }
$artifacts = @($files | ForEach-Object { [ordered]@{ name=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } })
$sourceUnchanged = (Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash -eq $payloadHash
$published = Join-Path $data 'published.md'
$publicationMatches = (Test-Path -LiteralPath $published -PathType Leaf) -and ((Get-FileHash -LiteralPath $published -Algorithm SHA256).Hash -eq $payloadHash)
[ordered]@{
    mode=$Mode
    exit_code=$process.ExitCode
    executable_bytes=(Get-Item -LiteralPath $exe).Length
    executable_sha256=$exeHash
    sampled_working_set_bytes=$ws
    sampled_private_bytes=$privateBytes
    sampled_cpu_ms=$cpuMs
    wall_ms=$timer.Elapsed.TotalMilliseconds
    samples=$samples
    requested_interval_ms=2
    source_unchanged=$sourceUnchanged
    publication_matches=$publicationMatches
    scope=$(if ($Mode -eq 'demo') { 'One Rust process: learn three sources, adapt, durable conformance, exact lab audit, store/load seed, replay and publish a local document.' } else { 'Separate Rust process: reopen models/journal and rebuild the adaptation entirely from cached responses, with no environment contact.' })
    limitations='Finite deterministic resettable interfaces. Sampled maxima are lower bounds; CPU omits the final unsampled interval. Wall includes launch/wait. OS cache not flushed. No power measurement.'
    data=$data
    artifacts=$artifacts
} | ConvertTo-Json -Depth 6 | Out-File -LiteralPath $report -NoClobber -Encoding utf8NoBOM
Get-Content -LiteralPath $report
if ($process.ExitCode -ne 0) { throw (Get-Content -LiteralPath $stderr -Raw) }
if (-not $sourceUnchanged -or -not $publicationMatches) { throw 'Document integrity check failed; measurements preserved.' }
