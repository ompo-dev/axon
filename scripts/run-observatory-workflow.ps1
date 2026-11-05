param(
    [ValidateSet('evolution','reactive','continuous','seed','controller','genome','language','cumulative','semantic','nursery','constructive','curriculum','integration','structural','tests')][string]$Mode = 'evolution',
    [ValidateRange(1,2147483647)][int]$Seed = 82641973,
    [ValidateRange(100,10000)][int]$Count = 1000,
    [ValidateSet('balanced','hybrid','mutation-only','no-inheritance','elitist','diverse','reinforced')][string]$Strategy = 'balanced',
    [string]$Curriculum = 'transfer',
    [ValidateSet('rust','rust-minimal','ui','check')][string]$Suite = 'rust',
    [ValidateSet('development','reserved')][string]$Validation = 'development',
    [ValidateRange(1024,65535)][int]$Port = 4317,
    [switch]$ResumeCompleted
)
$ErrorActionPreference = 'Stop'
$url = "http://127.0.0.1:$Port"
$state = Invoke-RestMethod "$url/api/state"
$body = [ordered]@{mode=$Mode;seed=$Seed}
if ($Mode -eq 'tests') { $body.suite = $Suite } else { $body.count = $Count }
if ($Mode -eq 'evolution') { $body.strategy = $Strategy; $body.curriculum = $Curriculum }
if ($Mode -eq 'curriculum') { $body.curriculum = if ($PSBoundParameters.ContainsKey('Curriculum')) { $Curriculum } else { 'mixed-128-v1' } }
if ($Mode -eq 'structural') {
    $body.Remove('seed')
    $body.Remove('count')
    $body.curriculum = if ($PSBoundParameters.ContainsKey('Curriculum')) { $Curriculum } else { 'compositional-v1' }
}
if ($Mode -eq 'constructive' -and ($PSBoundParameters.ContainsKey('Strategy') -or $PSBoundParameters.ContainsKey('Curriculum'))) {
    $body.strategy = if ($PSBoundParameters.ContainsKey('Strategy')) { $Strategy } else { 'elitist' }
    $body.curriculum = if ($PSBoundParameters.ContainsKey('Curriculum')) { $Curriculum } else { 'legacy' }
}
if ($Mode -in @('seed','controller','genome','language','cumulative','semantic')) { $body.validation=$Validation }
$previous = @()
if ($ResumeCompleted) {
    $previous = @($state.runs | Where-Object {
        $candidate = $_; $matches = $candidate.status -eq 'complete'
        foreach ($key in $body.Keys) { if ($candidate.options.$key -ne $body[$key]) { $matches = $false } }
        $matches
    })
    if ($previous.Count -gt 1) { throw 'Ambiguous repeated campaign; do not select a result by accuracy.' }
}
if ($previous.Count -eq 1) {
    $run = $previous[0]
    $content = (Invoke-WebRequest "$url/api/export?id=$($run.id)").Content
    if ($content -isnot [string]) { $content = [System.Text.Encoding]::UTF8.GetString($content) }
    $records = $content.Trim() -split "`n" | ForEach-Object { $_ | ConvertFrom-Json }
    if ($records[-1].event -ne 'end' -or $records[-1].status -ne 'complete' -or $records.Count -ne $run.trace.records) { throw 'Stored campaign lacks complete terminal evidence.' }
} else {
    $run = Invoke-RestMethod "$url/api/run" -Method Post -ContentType 'application/json' -Headers @{'X-Axon-Session'=$state.token} -Body ($body | ConvertTo-Json -Compress)
}
$deadline = [DateTime]::UtcNow.AddMinutes(11)
do {
    Start-Sleep -Milliseconds 100
    $state = Invoke-RestMethod "$url/api/state"
    $job = $state.runs | Where-Object { $_.id -eq $run.id } | Select-Object -First 1
    if (-not $job) { throw 'Run disappeared from the bounded ledger; inspect the observatory.' }
    if ([DateTime]::UtcNow -gt $deadline) { throw 'Waiting quota exceeded; process status remains in the observatory.' }
} while ($job.status -in @('running','paused','pausing','starting'))
$artifacts = Invoke-RestMethod "$url/api/artifacts?id=$($run.id)"
[ordered]@{
    id=$job.id; mode=$job.mode; options=$job.options; status=$job.status; exit_code=$job.exitCode
    elapsed_ms=($job.finished-$job.started); result=$job.result; cases=$job.cases
    candidates=$job.candidates; trace=$job.trace; artifacts=$artifacts.files; log_url="$url/api/export?id=$($job.id)"
} | ConvertTo-Json -Depth 12
if ($job.status -ne 'complete') { throw "Workflow ended as $($job.status); evidence remains in the ledger." }
