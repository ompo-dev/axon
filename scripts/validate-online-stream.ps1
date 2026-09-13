param(
    [string]$Directory = (Join-Path $PSScriptRoot '..\.ecc\benchmarks'),
    [string]$Pattern = 'online-stream-v2-*.json',
    [string]$Output
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$files = @(Get-ChildItem -LiteralPath $Directory -Filter $Pattern | Sort-Object Name)
if ($files.Count -eq 0) { throw 'No stream reports found' }
$rows = foreach ($file in $files) {
    $report = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
    if ($report.schema -ne 'axon-online-stream-v2' -or !$report.exact) { throw "Invalid report: $file" }
    $cases = $report.steps_per_size_phase * 32
    if ($report.observations -ne $cases -or $report.records.Count -ne 2 * $cases -or $report.exact_checks -ne 8 * $cases) {
        throw "Inconsistent counts: $file"
    }
    $sums = @(@([long]0, [long]0, [long]0, [long]0), @([long]0, [long]0, [long]0, [long]0))
    $positions = @{}
    foreach ($record in $report.records) {
        if ($record.Count -ne 10 -or $record[0] -notin @(0, 1) -or $record[1] -lt 0 -or $record[1] -gt 7 -or $record[3] -gt $record[2] -or $record[3] -lt 0 -or $record[4] -notin @(0, 1, 2, 3) -or $record[9] -notin @(0, 1)) {
            throw "Malformed sample: $file"
        }
        for ($candidate = 0; $candidate -lt 4; $candidate++) {
            $elapsed = [long]$record[5 + $candidate]
            if ($elapsed -lt 1) { throw "Nonpositive timing: $file" }
            $sums[$record[0]][$candidate] += $elapsed
        }
        $key = '{0}/{1}/{2}' -f $record[0], $record[1], $record[2]
        if (!$positions.ContainsKey($key)) { $positions[$key] = @(0, 0, 0, 0) }
        $positions[$key][$record[4]]++
    }
    if ($positions.Count -ne 64) { throw "Missing experimental contexts: $file" }
    foreach ($counts in $positions.Values) {
        foreach ($count in $counts) {
            if ($count -ne $report.steps_per_size_phase / 4) { throw "Unbalanced execution order: $file" }
        }
    }
    for ($candidate = 0; $candidate -lt 4; $candidate++) {
        if ($sums[0][$candidate] -ne $report.holdout.totals_ns[$candidate] -or $sums[1][$candidate] -ne $report.training.totals_ns[$candidate]) {
            throw "Timing totals do not match raw records: $file"
        }
    }
    $checkpoint = [IO.Path]::ChangeExtension($file.FullName, '.policy')
    if ((Get-Item -LiteralPath $checkpoint).Length -ne $report.checkpoint_bytes) { throw "Wrong checkpoint size: $file" }
    $train = $report.training.totals_ns
    $test = $report.holdout.totals_ns
    [pscustomobject]@{
        seed = $report.seed
        training_cases = $cases
        holdout_cases = $cases
        exact_checks = $report.exact_checks
        train_delta_ms = $train[0] / 1e6
        train_full_ms = $train[1] / 1e6
        train_heuristic_ms = $train[2] / 1e6
        train_learned_ms = $train[3] / 1e6
        holdout_delta_ms = $test[0] / 1e6
        holdout_full_ms = $test[1] / 1e6
        holdout_heuristic_ms = $test[2] / 1e6
        holdout_learned_ms = $test[3] / 1e6
        train_saving_vs_heuristic_percent = 100.0 * (1 - $train[3] / $train[2])
        holdout_saving_vs_heuristic_percent = 100.0 * (1 - $test[3] / $test[2])
        holdout_saving_vs_delta_percent = 100.0 * (1 - $test[3] / $test[0])
        holdout_saving_vs_full_percent = 100.0 * (1 - $test[3] / $test[1])
        combined_saving_vs_heuristic_percent = 100.0 * (1 - ($train[3] + $test[3]) / ($train[2] + $test[2]))
        experiment_wall_seconds = ($report.training_wall_ns + $report.holdout_wall_ns) / 1e9
    }
}
$summary = [pscustomobject]@{
    schema = 'axon-online-stream-validation-v1'
    reports = $files.Count
    raw_records = ($rows | Measure-Object -Property training_cases -Sum).Sum + ($rows | Measure-Object -Property holdout_cases -Sum).Sum
    exact_checks = ($rows | Measure-Object -Property exact_checks -Sum).Sum
    runs = @($rows)
}
$json = $summary | ConvertTo-Json -Depth 8
if ($Output) {
    $json | Set-Content -LiteralPath $Output -Encoding utf8
}
$json
