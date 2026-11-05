#Requires -Version 7.0
param(
    [string]$Binary = "target/release/axon.exe",
    [string]$OutputRoot = "target/text-learning-demo"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
function Resolve-TaskPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $root $Value))
}
$exe = Resolve-TaskPath $Binary
$output = Resolve-TaskPath $OutputRoot
$fixtures = Join-Path $root "examples/text-learning"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Build axon first: $exe" }
if (Test-Path -LiteralPath $output) { throw "Existing demo is preserved: $output" }
[IO.Directory]::CreateDirectory($output) | Out-Null
function Invoke-Axon([string[]]$Arguments) {
    $lines = @(& $exe @Arguments)
    if ($LASTEXITCODE -ne 0) { throw "AXON failed with exit code $LASTEXITCODE" }
    return $lines
}
$contacts = Join-Path $output "contacts-v1.axtext"
$inputPath = Join-Path $fixtures "contacts.input.csv"
$hashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash
Invoke-Axon @("text-fit", (Join-Path $fixtures "contacts.examples.csv"), "--name", "contact-reorder", "--context", "demo/contacts", "--skill", $contacts)
$preview = Invoke-Axon @("text-preview", $contacts, $inputPath, "--column", "contact", "--result-column", "normalized", "--context", "demo/contacts")
$preview
$expected = @(Import-Csv -LiteralPath (Join-Path $fixtures "contacts.expected.csv"))
foreach ($row in $expected) {
    if (-not ($preview | Where-Object { $_.Contains('output="' + $row.normalized + '"') })) { throw "Preview differs from the fixed demo expectation" }
}
# This fixed demonstration confirms only after comparing its preview with independent fixture outputs.
$token = ($preview | Where-Object { $_ -like "review_token=*" }) -replace '^review_token=', ''
if (@($token).Count -ne 1) { throw "No unambiguous preview token" }
$result = Join-Path $output "contacts.output.csv"
Invoke-Axon @("text-apply", $contacts, $inputPath, "--column", "contact", "--result-column", "normalized", "--context", "demo/contacts", "--output", $result, "--confirm", $token)
$actual = @(Import-Csv -LiteralPath $result)
if ($actual.Count -ne $expected.Count) { throw "Wrong result row count" }
for ($i = 0; $i -lt $expected.Count; $i++) { if ($actual[$i].normalized -cne $expected[$i].normalized) { throw "Wrong result at row $i" } }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash -ne $hashBefore) { throw "Original input changed" }
$old = Join-Path $output "surname-v1.axtext"
$new = Join-Path $output "surname-v2.axtext"
$names = Join-Path $fixtures "surname.input.csv"
Invoke-Axon @("text-fit", (Join-Path $fixtures "surname.examples.csv"), "--name", "surname", "--context", "demo/names", "--skill", $old)
$ambiguous = Invoke-Axon @("text-preview", $old, $names, "--column", "name", "--context", "demo/names")
$ambiguous
if (-not ($ambiguous | Where-Object { $_ -eq "needs_example=2" })) { throw "Expected ambiguous name observations" }
Invoke-Axon @("text-refine", $old, (Join-Path $fixtures "surname.refinement.csv"), "--context", "demo/names", "--skill", $new)
$refined = Invoke-Axon @("text-preview", $new, $names, "--column", "name", "--context", "demo/names")
$refined
if (-not ($refined | Where-Object { $_ -eq "agreed=2" })) { throw "Refinement did not resolve name extraction" }
[ordered]@{
    contacts_skill = $contacts
    contacts_output = $result
    contacts_checked_rows = $actual.Count
    old_skill = $old
    refined_skill = $new
    original_input_sha256 = $hashBefore
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash
    recorded_utc = [DateTime]::UtcNow.ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output "demo-report.json") -Encoding utf8
"Demo completed: $output"
