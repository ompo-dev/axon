param(
    [ValidateSet('lab','apply')][string]$Mode = 'apply',
    [string]$Bundle = '.axon-memory/demand-seed-demo-20261013-v2',
    [string]$OutputBase = '.ecc/benchmarks/demand-resource-20261013-v1'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$base = [System.IO.Path]::GetFullPath($OutputBase,$root)
$report = "$base.json"
$stdout = "$base.stdout.txt"
$stderr = "$base.stderr.txt"
$data = if ($Mode -eq 'lab') { "$base.outputs" } else { "$base.csv" }
foreach ($path in @($report,$stdout,$stderr,$data)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing existing output: $path" }
}
$exe = Join-Path $root $(if ($Mode -eq 'lab') { 'target/release/axon-uic-demand-lab.exe' } else { 'target/release/axon.exe' })
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build release binaries first.' }
if ($Mode -eq 'apply') {
    $bundlePath = [System.IO.Path]::GetFullPath($Bundle,$root)
    $seed = Join-Path $bundlePath 'seed.axseed'
    $inputCsv = Join-Path $bundlePath 'invoices.csv'
    $options = @('--from','invoice/raw','--goal','invoice/cents','--column','raw','--result-column','cents','--reduce','sum')
    $preview = & $exe seed-preview $seed $inputCsv @options
    if ($LASTEXITCODE -ne 0) { throw 'Preview failed.' }
    $tokenLine = @($preview | Where-Object { $_ -like 'review_token=*' })
    if ($tokenLine.Count -ne 1) { throw 'Missing review token.' }
    $token = $tokenLine[0].Substring('review_token='.Length)
    $arguments = @('seed-apply',"`"$seed`"","`"$inputCsv`"") + $options + @('--output',"`"$data`"",'--confirm',$token)
} else { $arguments = @("`"$data`"") }
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
if ($process.ExitCode -ne 0) { throw (Get-Content -LiteralPath $stderr -Raw) }
$files = if ($Mode -eq 'lab') { @('seed.axseed','report.txt','measurements.csv','consolidated.csv','cursor-final.axcursor') | ForEach-Object { Get-Item -LiteralPath (Join-Path $data $_) } } else { @(Get-Item -LiteralPath $data) }
$artifacts = @($files | ForEach-Object { [ordered]@{ name=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } })
[ordered]@{
    mode=$Mode
    executable_bytes=(Get-Item -LiteralPath $exe).Length
    executable_sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    sampled_working_set_bytes=$ws
    sampled_private_bytes=$privateBytes
    sampled_cpu_ms=$cpuMs
    wall_ms=$timer.Elapsed.TotalMilliseconds
    samples=$samples
    requested_interval_ms=2
    scope=$(if ($Mode -eq 'lab') { 'One Rust process: all training, storage, 18 benchmark batches, reference interpretation, CLI preview/publication and resumed behavior monitor. Not a pure inference RSS comparison.' } else { 'One Rust process: seed load, CSV read, composed inference, exact reduction and durable publication. Preview runs before the timed process. Logical cold modules; OS cache not flushed.' })
    limitations='Sampled maxima are lower bounds, not exact peaks. CPU excludes final unsampled interval; wall includes process launch/wait. No power measurement.'
    data=$data
    artifacts=$artifacts
} | ConvertTo-Json -Depth 6 | Out-File -LiteralPath $report -NoClobber -Encoding utf8NoBOM
Get-Content -LiteralPath $report
