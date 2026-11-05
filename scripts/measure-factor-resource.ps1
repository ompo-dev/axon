param(
    [ValidateSet('demo','reuse')][string]$Mode = 'demo',
    [string]$Inputs = 'examples/factor-documents',
    [string]$Bundle = '.axon-memory/factor-demo-20261013-v1/models/factors.axseed',
    [string]$OutputBase = '.ecc/benchmarks/factor-resource-20261013-v1'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$base = [System.IO.Path]::GetFullPath($OutputBase,$root)
$inputPath = [System.IO.Path]::GetFullPath($Inputs,$root)
$data = "$base.outputs"
$report = "$base.json"
$stdout = "$base.stdout.txt"
$stderr = "$base.stderr.txt"
foreach ($p in @($data,$report,$stdout,$stderr)) { if (Test-Path -LiteralPath $p) { throw "Refusing existing output: $p" } }
$exe = Join-Path $root 'target/release/axon.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the release CLI first.' }
$sources = @(Get-ChildItem -LiteralPath $inputPath -File -Filter '*.md' | Sort-Object Name | ForEach-Object { [ordered]@{ path=$_.FullName; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } })
if ($sources.Count -lt 2 -or $sources.Count -gt 5) { throw 'Expected 2..5 Markdown sources.' }
$arguments = if ($Mode -eq 'demo') { @('factor-demo',"`"$data`"",'--inputs',"`"$inputPath`"") } else {
    $bundlePath = [System.IO.Path]::GetFullPath($Bundle,$root)
    @('factor-publish',"`"$bundlePath`"","`"$data`"",'--inputs',"`"$inputPath`"")
}
$exeHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
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
$unchanged = $true
$publicationMatches = $true
foreach ($source in $sources) {
    $unchanged = $unchanged -and ((Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash -eq $source.sha256)
}
$mappings = @(Get-Content -LiteralPath $stdout | Where-Object { $_ -match '^document_\d+_source=' })
$publicationMatches = $mappings.Count -eq $sources.Count
foreach ($mapping in $mappings) {
    if ($mapping -notmatch '^document_(\d+)_source=(.*)$') { $publicationMatches = $false; continue }
    $index = [int]$Matches[1]
    $sourcePath = $Matches[2]
    $sourceRecord = @($sources | Where-Object { $_.path -eq $sourcePath })
    $published = Join-Path $data $(if ($Mode -eq 'demo') { "published/document-$index.md" } else { "document-$index.md" })
    $publicationMatches = $publicationMatches -and $sourceRecord.Count -eq 1 -and (Test-Path -LiteralPath $published -PathType Leaf) -and ((Get-FileHash -LiteralPath $published -Algorithm SHA256).Hash -eq $sourceRecord[0].sha256)
}
$files = if (Test-Path -LiteralPath $data -PathType Container) { @(Get-ChildItem -LiteralPath $data -File -Recurse) } else { @() }
$artifacts = @($files | ForEach-Object { [ordered]@{ name=$_.FullName.Substring($data.Length+1); bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } })
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
    sources_unchanged=$unchanged
    publications_match=$publicationMatches
    sources=$sources
    data=$data
    scope=$(if ($Mode -eq 'demo') { 'Whole Rust process: shared observation cache, projected learning, whole-product lab audit, catalog/model save/reload, joint planning and real multi-file publication.' } else { 'Separate Rust process: load existing learned models, plan and publish payloads with actual environment actions; no new learning queries.' })
    limitations='Finite discrete deterministic resettable adapter. Supplied schema/goal tokens. Cache is in RAM, not a durable experience journal. Sampled resource maxima are lower bounds; CPU omits last interval. Wall includes launch/wait. OS cache not flushed. No power measurement.'
    artifacts=$artifacts
} | ConvertTo-Json -Depth 6 | Out-File -LiteralPath $report -NoClobber -Encoding utf8NoBOM
Get-Content -LiteralPath $report
if ($process.ExitCode -ne 0) { throw (Get-Content -LiteralPath $stderr -Raw) }
if (-not $unchanged -or -not $publicationMatches) { throw 'Integrity failure; measurements preserved.' }
