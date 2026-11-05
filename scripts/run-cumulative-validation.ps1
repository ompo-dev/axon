param(
    [string]$Url = 'http://127.0.0.1:4319',
    [ValidateRange(1,2147483647)][long]$Seed = 39172841,
    [ValidateRange(100,10000)][int]$Count = 10000,
    [ValidateSet('development','reserved')][string]$Validation = 'development'
)
$ErrorActionPreference = 'Stop'
function Response-Text($response) {
    if ($response.Content -is [byte[]]) { return [Text.Encoding]::UTF8.GetString($response.Content) }
    return [string]$response.Content
}
$uri = [Uri]$Url
if ($uri.Scheme -ne 'http' -or $uri.Host -ne '127.0.0.1' -or $uri.AbsolutePath -ne '/') { throw 'Only the local observer is allowed' }
$state = Invoke-RestMethod "$Url/api/state"
if ($state.current.status -in @('running','starting','pausing','paused')) { throw 'Another run is active; it was not interrupted' }
$body = @{mode='cumulative';seed=$Seed;count=$Count;validation=$Validation} | ConvertTo-Json -Compress
$run = Invoke-RestMethod "$Url/api/run" -Method Post -ContentType 'application/json' -Headers @{'X-Axon-Session'=$state.token} -Body $body
$deadline = [DateTime]::UtcNow.AddMinutes(11)
do {
    Start-Sleep -Milliseconds 500
    $state = Invoke-RestMethod "$Url/api/state"
    if ($state.current.id -ne $run.id) { throw 'Current run changed; no other run was touched' }
    if ([DateTime]::UtcNow -gt $deadline) { throw 'Observer deadline exceeded' }
} while ($state.current.status -in @('running','starting','pausing','paused'))
$job = $state.current
if ($job.status -ne 'complete' -or $job.exitCode -ne 0) { throw "Run failed: $($job.output)" }
if ($job.trace.gaps -ne 0 -or $job.result.checked -ne $Count*32 -or $job.result.wrong -ne 0) { throw 'Core accounting failed' }
$csv = Invoke-WebRequest "$Url/api/artifact?id=$($job.id)&name=results.csv"
$rows = @((Response-Text $csv) | ConvertFrom-Csv)
if ($rows.Count -ne 32) { throw 'Missing paired comparisons' }
foreach ($row in $rows) {
    $total = [long]$row.correct + [long]$row.domain_refusals + [long]$row.unavailable + [long]$row.wrong
    if ($total -ne $Count -or [long]$row.heldout -ne $Count) { throw 'Missing scored examples' }
}
$export = Invoke-WebRequest "$Url/api/export?id=$($job.id)"
$records = @((Response-Text $export).Trim() -split '\r?\n' | ForEach-Object { $_ | ConvertFrom-Json })
$core = @($records | Where-Object { $null -ne $_.coreSeq })
for ($i=0;$i -lt $core.Count;$i++) { if ($core[$i].coreSeq -ne $i+1) { throw 'Core sequence gap' } }
if ($records.Count -ne $job.trace.records -or $records[-1].event -ne 'end') { throw 'Incomplete durable trace' }
$job | Select-Object id,mode,status,exitCode,trace,result | ConvertTo-Json -Depth 8
