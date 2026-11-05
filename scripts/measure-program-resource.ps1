param(
    [Parameter(Mandatory=$true)][string]$OutputBase,
    [UInt64]$Seed = 88603179,
    [int]$Rows = 20000,
    [ValidateSet('lab','evaluate','abstraction','continuous','reactive','evolution','controller-replay','curriculum-run','curriculum-exercise','curriculum-structural','curriculum-algorithm-batch','curriculum-encoder-batch')][string]$Mode = 'lab',
    [ValidateSet('hybrid','mutation-only','no-inheritance','balanced')][string]$EvolutionMode = 'hybrid',
    [ValidateSet('core','transfer')][string]$EvolutionSuite = 'core',
    [string]$Sources,
    [string]$Skill,
    [string]$Evaluation,
    [string]$Context,
    [string]$ControllerBundle,
    [string]$ControllerStream,
    [ValidateRange(1,2147483647)][int]$ControllerRepair = 1,
    [ValidateSet('stack','shared')][string]$Backend = 'stack',
    [string]$Executable,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if ($Executable -and -not $NoBuild) { throw 'An explicit frozen executable requires NoBuild.' }
if ($Mode -eq 'continuous') { if ($Rows -lt 1000 -or $Rows -gt 1000000) { throw 'Continuous observations must be 1000..1000000.' } }
elseif ($Rows -lt 100 -or $Rows -gt 100000) { throw 'Rows must be 100..100000.' }
if (-not $NoBuild) {
    Push-Location $root
    $extra = if ($Mode -like 'curriculum-*') { @('--features','curriculum','--bin','axon-uic-curriculum-lab') } else { @() }
    try { & cargo build --release @extra --bin axon --bin axon-uic-program-lab --bin axon-uic-abstraction-lab --bin axon-uic-continuous-lab --bin axon-uic-live-agent --bin axon-uic-reactive-lab --bin axon-uic-reactive-agent --bin axon-uic-evolution-lab --bin axon-uic-controller-lab; if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' } }
    finally { Pop-Location }
}
$base = if ([IO.Path]::IsPathRooted($OutputBase)) { [IO.Path]::GetFullPath($OutputBase) } else { [IO.Path]::GetFullPath((Join-Path $root $OutputBase)) }
$isBatch = $Mode -in @('curriculum-algorithm-batch','curriculum-encoder-batch')
$output = if ($Mode -eq 'controller-replay') { "$base.csv" } elseif ($isBatch) { "$base.predictions.json" } else { "$base.outputs" }
$stdout = "$base.stdout.log"
$stderr = "$base.stderr.log"
$report = "$base.json"
foreach ($path in @($output,$stdout,$stderr,$report)) { if (Test-Path -LiteralPath $path) { throw "Output already exists: $path" } }
if ($isBatch -and (Test-Path -LiteralPath "$base.birth")) { throw 'Birth output already exists.' }
[IO.Directory]::CreateDirectory((Split-Path -Parent $base)) | Out-Null
$binary = Join-Path $root $(if ($Mode -like 'curriculum-*') { 'target/release/axon-uic-curriculum-lab.exe' } elseif ($Mode -eq 'lab') { 'target/release/axon-uic-program-lab.exe' } elseif ($Mode -eq 'abstraction') { 'target/release/axon-uic-abstraction-lab.exe' } elseif ($Mode -eq 'continuous') { 'target/release/axon-uic-continuous-lab.exe' } elseif ($Mode -eq 'reactive') { 'target/release/axon-uic-reactive-lab.exe' } elseif ($Mode -eq 'evolution') { 'target/release/axon-uic-evolution-lab.exe' } elseif ($Mode -eq 'controller-replay') { 'target/release/axon-uic-controller-lab.exe' } else { 'target/release/axon.exe' })
if ($Executable) { $binary = [IO.Path]::GetFullPath($Executable,$root) }
$arguments = if ($Mode -eq 'curriculum-structural') {
    if (-not $Sources) { throw 'Structural curriculum requires Sources JSON.' }
    @('structure',"`"$Sources`"","`"$output`"")
} elseif ($isBatch) {
    if (-not $Sources -or -not $Evaluation) { throw 'Batch requires Sources number and Evaluation requests.' }
    $command = if ($Mode -eq 'curriculum-encoder-batch') { 'encoder-batch' } else { 'algorithm-batch' }
    @($command,"`"$Sources`"","`"$base.birth`"","`"$Evaluation`"","`"$output`"")
} elseif ($Mode -eq 'curriculum-exercise') { @('exercise',"`"$output`"") } elseif ($Mode -eq 'curriculum-run') {
    if (-not $Sources) { throw 'Curriculum run requires prepared Sources JSON.' }
    [IO.Directory]::CreateDirectory($output) | Out-Null
    @('run',"`"$Sources`"","`"$output`"",$Seed)
} elseif ($Mode -eq 'lab') { @("`"$output`"",$Seed,$Rows) } elseif ($Mode -eq 'abstraction') {
    if (-not $Sources) { throw 'Abstraction mode requires Sources.' }
    @("`"$Sources`"","`"$output`"",$Seed,$Rows)
} elseif ($Mode -eq 'evolution') { @("`"$output`"",$Seed,$Rows,$EvolutionMode,$EvolutionSuite) } elseif ($Mode -in @('continuous','reactive')) { @("`"$output`"",$Seed,$Rows) } elseif ($Mode -eq 'controller-replay') {
    if (-not $ControllerBundle -or -not $ControllerStream) { throw 'Controller replay requires bundle and stream.' }
    $bundle = [IO.Path]::GetFullPath($ControllerBundle,$root)
    if ([IO.Path]::GetFullPath((Split-Path -Parent $output)) -ne $bundle) { throw 'Controller outputs must stay in the named bundle.' }
    if ($ControllerStream -notmatch '^[a-zA-Z0-9_-]+\.tsv$') { throw 'Controller stream must be a plain sibling TSV filename.' }
    @('replay',"`"$bundle`"",$ControllerStream,[IO.Path]::GetFileName($output),$Seed,$ControllerRepair)
} else {
    if (-not $Skill -or -not $Evaluation -or -not $Context) { throw 'Evaluation requires Skill, Evaluation and Context.' }
    @('numeric-eval',"`"$Skill`"","`"$Evaluation`"",'--context',"`"$Context`"",'--backend',$Backend)
}
$binaryHash = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
$timer = [Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $binary -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$samples = 0; $working = 0L; $private = 0L; $osPeak = 0L; $cpu = $null
while (-not $process.HasExited) {
    try {
        $process.Refresh()
        $working = [Math]::Max($working,$process.WorkingSet64)
        $private = [Math]::Max($private,$process.PrivateMemorySize64)
        $osPeak = [Math]::Max($osPeak,$process.PeakWorkingSet64)
        $cpu = $process.TotalProcessorTime.TotalMilliseconds
        $samples++
    } catch { }
    Start-Sleep -Milliseconds 2
}
$process.WaitForExit(); $timer.Stop()
$result = [ordered]@{
    mode=$Mode; backend=$(if ($Mode -eq 'evaluate') { $Backend } else { $null }); evolution_mode=$(if ($Mode -eq 'evolution') { $EvolutionMode } else { $null }); seed=$(if ($Mode -in @('curriculum-exercise','curriculum-structural') -or $isBatch) { $null } else { $Seed }); requested_lab_rows=$(if ($Mode -notin @('evaluate','controller-replay','curriculum-run','curriculum-exercise','curriculum-structural') -and -not $isBatch) { $Rows } else { $null });
    evolution_suite=$(if ($Mode -eq 'evolution') { $EvolutionSuite } else { $null });
    output=$(if ($Mode -ne 'evaluate') { $output } else { $null }); sources=$Sources; skill=$Skill; evaluation=$Evaluation; context=$Context; exit_code=$process.ExitCode;
    controller_bundle=$(if ($Mode -eq 'controller-replay') { $bundle } else { $null }); controller_stream=$(if ($Mode -eq 'controller-replay') { $ControllerStream } else { $null }); controller_repair=$(if ($Mode -eq 'controller-replay') { $ControllerRepair } else { $null });
    lifecycle_ms=$timer.Elapsed.TotalMilliseconds; samples=$samples; requested_sample_interval_ms=2;
    max_sampled_working_set_bytes=$(if ($samples -gt 0) { $working } else { $null }); max_sampled_private_bytes=$(if ($samples -gt 0) { $private } else { $null });
    max_observed_os_peak_working_set_bytes=$(if ($samples -gt 0) { $osPeak } else { $null }); cpu_ms_last_sample=$cpu;
    binary_bytes=(Get-Item -LiteralPath $binary).Length; binary_sha256=$binaryHash; binary_path=$binary;
    metric_scope=$(if ($Mode -eq 'controller-replay') { 'One second-process replay: bundle admission replay, input buffering, demand loading, two learned procedures, fixed-route comparison, simulated device relearning and CSV output. Excludes Bun/UI and the lab parent; not pure inference or energy.' } elseif ($Mode -eq 'curriculum-encoder-batch') { 'Entire second-process AXON CLI: numeric birth, JSON inputs, demand-loaded encoder, optional byte computation and predictions IO. Excludes Bun/UI, parent/oracle, training and energy. Sampling may miss final intervals; CPU is only the last sampled cumulative value, not guaranteed final CPU. Not pure inference, isolated model memory or a fair cross-hardware benchmark.' } else { 'Entire selected process. Lab includes search, data generation, validation and file IO. Evaluation includes the full AXON CLI, capsule and input loading. Memory observations can miss the final interval; no energy measurement.' });
    stdout=$stdout; stderr=$stderr
}
$json = $result | ConvertTo-Json -Depth 5
$file = [IO.File]::Open($report,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
$writer = [IO.StreamWriter]::new($file,[Text.UTF8Encoding]::new($false))
try { $writer.Write($json) } finally { $writer.Dispose() }
$json
if ($process.ExitCode -ne 0) { throw "Program lab failed; see $stderr and $stdout" }
