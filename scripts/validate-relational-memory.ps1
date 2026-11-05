param(
    [string]$Binary = "target/release/axon.exe",
    [string]$Memory = ".axon-memory/knowledge.axm",
    [string]$Evaluation = "evals/knowledge_relational_dev.eval",
    [string]$Output = "target/relational-memory.json"
)

$ErrorActionPreference = "Stop"
$taskRoot = Split-Path -Parent $PSScriptRoot

function Resolve-TaskPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $taskRoot $Value))
}

$binaryPath = Resolve-TaskPath $Binary
$memoryPath = Resolve-TaskPath $Memory
$evaluationPath = Resolve-TaskPath $Evaluation
$outputPath = Resolve-TaskPath $Output
foreach ($inputPath in @($binaryPath, $memoryPath, $evaluationPath)) {
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "Missing input: $inputPath"
    }
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Report already exists: $outputPath"
}

$memoryHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $memoryPath).Hash
$timer = [Diagnostics.Stopwatch]::StartNew()
$rawOutput = @(& $binaryPath memory-coherence-eval $evaluationPath --memory $memoryPath --limit 5)
$exitCode = $LASTEXITCODE
$timer.Stop()
if ($exitCode -ne 0) {
    throw "Evaluation failed with exit code $exitCode"
}
$memoryHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $memoryPath).Hash
if ($memoryHashBefore -ne $memoryHashAfter) {
    throw "Evaluation modified the memory snapshot"
}

$metrics = [ordered]@{}
foreach ($line in $rawOutput) {
    if ($line -match '^(cases|positive_cases|negative_cases|top1|recall_at_5|correct_abstentions|overall_correct|mrr|mean_top_path_coverage|mean_top_path_coherence)=(.*)$') {
        $metrics[$Matches[1]] = $Matches[2]
    }
}
$report = [ordered]@{
    protocol = "axon-relational-memory-v1"
    recorded_utc = [DateTime]::UtcNow.ToString("o")
    evaluation = $evaluationPath
    evaluation_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $evaluationPath).Hash
    memory = $memoryPath
    memory_sha256 = $memoryHashBefore
    memory_bytes = (Get-Item -LiteralPath $memoryPath).Length
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $binaryPath).Hash
    executable_bytes = (Get-Item -LiteralPath $binaryPath).Length
    elapsed_ms = [Math]::Round($timer.Elapsed.TotalMilliseconds, 3)
    metrics = $metrics
    raw_output = $rawOutput
}
$parentPath = Split-Path -Parent $outputPath
[IO.Directory]::CreateDirectory($parentPath) | Out-Null
$json = $report | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($outputPath, $json, [Text.UTF8Encoding]::new($false))
$json
